"""Regression tests for integrity fixes (audio IO, splits, labels, metrics).

Each test pins a fixed breakage so it cannot silently regress:
  - audio IO must not require the TorchCodec backend (soundfile only).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sine(freq: float = 440.0, sr: int = 16000, dur_s: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(sr * dur_s), dtype=torch.float32) / sr
    return torch.sin(2 * torch.pi * freq * t)


class TestAudioIO:
    def test_roundtrip_mono_16k(self, tmp_path):
        from data.audio_io import read_mono, write_wav

        wav = _sine()
        p = tmp_path / "mono.wav"
        write_wav(str(p), wav, 16000)
        out = read_mono(str(p))
        assert out.shape == (16000,)
        assert out.dtype == torch.float32
        assert torch.allclose(out, wav, atol=1e-3)

    def test_stereo_44k_resamples_to_mono_16k(self):
        from data.audio_io import read_mono, write_wav

        stereo = torch.stack([_sine(sr=44100), _sine(freq=880.0, sr=44100)])
        buf = io.BytesIO()
        write_wav(buf, stereo, 44100)
        buf.seek(0)
        out = read_mono(buf)
        assert out.dim() == 1
        assert out.shape == (16000,)

    def test_truncates_to_max_len(self, tmp_path):
        from data.audio_io import read_mono, write_wav

        wav = _sine(dur_s=12.0)
        p = tmp_path / "long.wav"
        write_wav(str(p), wav, 16000)
        out = read_mono(str(p), max_len_s=10.0)
        assert out.shape == (160000,)

    def test_no_torchcodec_dependency(self):
        """torchaudio.load/save must not be used anywhere in live code."""
        root = Path(__file__).resolve().parent.parent
        offenders = []
        for sub in ("data", "models", "serve", "scripts", "training", "evaluation"):
            for py in (root / sub).rglob("*.py"):
                if py.name == "audio_io.py":
                    continue  # the sanctioned IO layer (uses soundfile, not torchaudio)
                text = py.read_text()
                if "torchaudio.load" in text or "torchaudio.save" in text:
                    offenders.append(str(py.relative_to(root)))
        assert offenders == []

    def test_load_audio_entrypoint(self, tmp_path):
        from data.datasets import load_audio

        from data.audio_io import write_wav

        p = tmp_path / "x.wav"
        write_wav(str(p), _sine(sr=8000, dur_s=0.5), 8000)
        out = load_audio(str(p))
        assert out.shape == (8000,)  # 0.5s @ 16kHz


class TestProsodyLookups:
    def _write_stats(self, tmp_path, stem="prosody_stats"):
        import json

        base = tmp_path / f"{stem}.json"
        base.write_text(json.dumps({"__global__": {"mean": [0, 0, 0], "std": [1, 1, 1]}}))
        return base

    def test_split_files_preferred(self, tmp_path):
        import json

        from data.datasets import load_prosody_lookups

        base = self._write_stats(tmp_path)
        (tmp_path / "prosody_stats.train.zscores.json").write_text(json.dumps({"a": [0.1, 0.2, 0.3]}))
        (tmp_path / "prosody_stats.val.zscores.json").write_text(json.dumps({"b": [0.4, 0.5, 0.6]}))
        train_lut, val_lut, used_legacy = load_prosody_lookups(base)
        assert used_legacy is False
        assert set(train_lut) == {"a"}
        assert set(val_lut) == {"b"}

    def test_legacy_combined_flagged(self, tmp_path):
        import json

        from data.datasets import load_prosody_lookups

        base = self._write_stats(tmp_path)
        (tmp_path / "prosody_stats.zscores.json").write_text(json.dumps({"a": [0.1, 0.2, 0.3]}))
        train_lut, val_lut, used_legacy = load_prosody_lookups(base)
        assert used_legacy is True
        assert train_lut is val_lut  # same contaminated object for both splits

    def test_missing_files_return_nones(self, tmp_path):
        from data.datasets import load_prosody_lookups

        base = self._write_stats(tmp_path)
        train_lut, val_lut, used_legacy = load_prosody_lookups(base)
        assert train_lut is None and val_lut is None and used_legacy is False


class TestHonestMetrics:
    def test_empty_class_reported_with_ceiling(self):
        import numpy as np

        from evaluation.metrics import compute_all_metrics

        # Deception column has zero positives (current repo-wide reality)
        labels = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
        probs = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.1]])
        metrics = compute_all_metrics(probs, labels)
        assert metrics["empty_classes"] == ["deception"]
        assert "ap_deception" not in metrics  # undefined without positives
        # Macro-F1 over a 3-type head with one empty class caps at 2/3
        assert metrics["macro_f1"] <= 2 / 3 + 1e-9

    def test_mustard_only_ceiling_is_one_third(self):
        """Numerically executes the critic's 1/3 ceiling: MUSTARD emits only
        the sarcasm column, so macro-F1 over 3 heads caps at 1/3."""
        import numpy as np

        from evaluation.metrics import compute_all_metrics

        labels = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
        probs = labels.astype(float)  # perfect predictions
        metrics = compute_all_metrics(probs, labels)
        assert metrics["macro_f1"] == 1 / 3
        assert metrics["empty_classes"] == ["suppression", "deception"]

    def test_print_metrics_handles_non_floats(self, capsys):
        from evaluation.metrics import print_metrics

        print_metrics({"macro_f1": 0.5, "empty_classes": ["deception"], "n": 4})
        out = capsys.readouterr().out
        assert "deception" in out and "0.5000" in out


class _MockAudioEncoder(torch.nn.Module):
    output_dim = 768

    def forward(self, audio, attention_mask=None, return_frames=False):
        pooled = torch.randn(audio.size(0), self.output_dim, device=audio.device)
        if return_frames:
            return pooled, torch.randn(audio.size(0), 50, self.output_dim, device=audio.device)
        return pooled


class _MockTextEncoder(torch.nn.Module):
    output_dim = 1024

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input_ids, attention_mask, return_tokens=False):
        B = input_ids.shape[0]
        pooled = torch.randn(B, self.output_dim, device=input_ids.device)
        if return_tokens:
            return pooled, torch.randn(B, input_ids.shape[1], self.output_dim, device=input_ids.device)
        return pooled


class TestFailClosedLabels:
    @staticmethod
    def _tiny_model():
        import unittest.mock

        from models import conflictnet as cn_module
        from models.conflictnet import ConflictNet

        patchers = [
            unittest.mock.patch.object(cn_module, "build_audio_encoder", return_value=_MockAudioEncoder()),
            unittest.mock.patch.object(cn_module, "DeBERTaEncoder", _MockTextEncoder),
        ]
        for p in patchers:
            p.start()
        try:
            return ConflictNet(
                audio_encoder_name="wavlm", embed_dim=32, n_conflict_types=3,
                use_speaker_norm=False, use_temporal=False, use_word_divergence=False,
                use_swap_pretraining=True, use_cross_attn_injection=False,
                use_speaker_adaptive_threshold=False, use_baseline_subtract=False,
                lora_r=0, temporal_n_layers=1, temporal_n_heads=2, temporal_max_turns=4,
            )
        finally:
            for p in patchers:
                p.stop()

    def test_missing_mask_skips_subtype_loss(self):
        model = self._tiny_model()
        model.eval()
        audio = torch.randn(2, 16000)
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        type_labels = torch.ones(2, 3)  # proxy-looking labels
        with torch.no_grad():
            out = model(
                audio=audio, input_ids=input_ids, attention_mask=attention_mask,
                conflict_type_labels=type_labels, has_real_type_labels=None,
            )
        assert out.loss is not None  # contrastive still applies
        assert out.loss_breakdown["type_loss"] == 0.0  # subtype skipped, not trained

    def test_pretraining_skips_supervised_losses(self):
        model = self._tiny_model()
        model.eval()
        audio = torch.randn(2, 16000)
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        with torch.no_grad():
            out = model(
                audio=audio, input_ids=input_ids, attention_mask=attention_mask,
                conflict_type_labels=torch.ones(2, 3),
                severity_labels=torch.rand(2, 1),
                has_real_type_labels=torch.ones(2, dtype=torch.bool),
                pretraining=True,
            )
        assert out.loss is not None  # contrastive + swap
        assert out.loss_breakdown["type_loss"] == 0.0
        assert out.loss_breakdown["severity_mse"] == 0.0


class TestTextgridFallback:
    _TG = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 2.5
tiers? <exists>
size = 2
item [1]:
    class = "IntervalTier"
    name = "phones"
    xmin = 0
    xmax = 2.5
    intervals: size = 1
    intervals [1]:
        xmin = 0
        xmax = 2.5
        text = "h eh l ow"
item [2]:
    class = "IntervalTier"
    name = "words"
    xmin = 0
    xmax = 2.5
    intervals: size = 3
    intervals [1]:
        xmin = 0
        xmax = 1.0
        text = "hello"
    intervals [2]:
        xmin = 1.0
        xmax = 1.5
        text = ""
    intervals [3]:
        text = "world"
        xmax = 2.5
        xmin = 1.5
"""

    def test_parses_standard_and_reordered_intervals(self, tmp_path):
        from scripts.run_mfa_alignment import _parse_textgrid_fallback

        p = tmp_path / "utt.TextGrid"
        p.write_text(self._TG)
        words = _parse_textgrid_fallback(str(p))
        assert [(w["word"], w["start"], w["end"]) for w in words] == [
            ("hello", 0.0, 1.0),
            ("world", 1.5, 2.5),
        ]


class TestStats:
    def _toy(self, n=120, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        labels = (rng.random((n, 3)) < 0.4).astype(float)
        probs = np.clip(labels + rng.normal(0, 0.35, (n, 3)), 0, 1)
        return probs, labels

    def test_bootstrap_ci_brackets_point(self):
        from scripts.stats import bootstrap_ci

        probs, labels = self._toy()
        rep = bootstrap_ci(probs, labels, "macro_f1", n_bootstrap=500, seed=7)
        assert rep["ci_lo"] <= rep["point"] <= rep["ci_hi"]
        assert rep["se"] > 0 and rep["n_samples"] == 120

    def test_bootstrap_is_deterministic(self):
        from scripts.stats import bootstrap_ci

        probs, labels = self._toy()
        a = bootstrap_ci(probs, labels, "binary_f1", n_bootstrap=200, seed=3)
        b = bootstrap_ci(probs, labels, "binary_f1", n_bootstrap=200, seed=3)
        assert a == b

    def test_paired_bootstrap_detects_no_difference(self):
        from scripts.stats import paired_bootstrap_pvalue

        probs, labels = self._toy()
        rep = paired_bootstrap_pvalue(probs, probs.copy(), labels, n_bootstrap=500, seed=1)
        assert rep["delta"] == 0.0
        assert rep["p_value"] > 0.05 and rep["significant_05"] is False

    def test_mcnemar_identical_models(self):
        import numpy as np

        from scripts.stats import mcnemar_test

        y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        rep = mcnemar_test(y, y.copy(), y.copy())
        assert rep["p_value"] == 1.0 and rep["significant_05"] is False

    def test_mcnemar_detects_strong_difference(self):
        import numpy as np

        from scripts.stats import mcnemar_test

        # A correct on all 30, B wrong on all 30 → maximally significant
        y = np.tile([1, 0], 15)
        rep = mcnemar_test(y, y.copy(), 1 - y)
        assert rep["p_value"] < 0.05 and rep["significant_05"] is True


class TestAdaptiveThreshold:
    def test_offset_range_is_symmetric(self):
        from models.classifier import SpeakerAdaptiveThreshold

        sat = SpeakerAdaptiveThreshold(embed_dim=16, max_offset=0.3)
        offsets = sat(torch.randn(64, 16))
        assert bool((offsets >= -0.3).all()) and bool((offsets <= 0.3).all())

    def test_distinct_speakers_give_distinct_offsets(self):
        from models.classifier import SpeakerAdaptiveThreshold

        sat = SpeakerAdaptiveThreshold(embed_dim=8, max_offset=0.3)
        off_e = sat(torch.ones(1, 8)).item()
        off_m = sat(-torch.ones(1, 8)).item()
        assert off_e != off_m


class TestFailClosedEncoders:
    """Training must refuse silent random-init encoder fallbacks."""

    def test_refuses_text_fallback(self):
        import pytest

        from models.encoders.text import _TextFallbackEncoder
        from scripts.train import refuse_fallback_encoders

        model = torch.nn.Sequential(_TextFallbackEncoder(vocab_size=128, embed_dim=16))
        with pytest.raises(SystemExit, match="REFUSING"):
            refuse_fallback_encoders(model)

    def test_refuses_spectrogram_fallback(self):
        import pytest

        from models.encoders.audio import _SpectrogramEncoder
        from scripts.train import refuse_fallback_encoders

        model = torch.nn.Sequential(_SpectrogramEncoder(output_dim=16))
        with pytest.raises(SystemExit, match="REFUSING"):
            refuse_fallback_encoders(model)

    def test_allows_clean_model(self):
        from scripts.train import refuse_fallback_encoders

        model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
        assert refuse_fallback_encoders(model) is None

    def test_allow_flag_defaults_off(self):
        from scripts.train import parse_args

        assert parse_args([]).allow_fallback_encoders is False
        assert parse_args(["--allow_fallback_encoders"]).allow_fallback_encoders is True
