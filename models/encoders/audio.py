"""Audio encoders: Emotion2Vec, WavLM, wav2vec2.

Each returns (batch, encoder_dim) from raw waveform input.
All encoders accept an optional attention_mask for padded audio.
When HuggingFace models are unavailable (e.g. Kaggle without internet),
falls back to a simple spectrogram-based encoder.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _SpectrogramEncoder(nn.Module):
    """Simple spectrogram + CNN audio encoder fallback (no external models needed)."""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("hann", torch.hann_window(512))
        self.conv = nn.Sequential(
            nn.Conv1d(257, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Conv1d(256, output_dim, kernel_size=1),
        )

    def forward(self, audio, attention_mask=None, return_frames=False):
        spec = torch.stft(audio, n_fft=512, hop_length=160, window=self.hann,
                          return_complex=True).abs()
        pooled = self.conv(spec).mean(dim=-1)
        if return_frames:
            frames = self.conv(spec).permute(0, 2, 1)
            return pooled, frames
        return pooled


class Wav2Vec2Encoder(nn.Module):
    """Baseline audio encoder from HuBERT-CLAP, falls back to spectrogram."""

    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h", freeze: bool = True):
        super().__init__()
        self._encoder = None
        self.output_dim = 768
        try:
            from transformers import Wav2Vec2Model
            enc = Wav2Vec2Model.from_pretrained(model_name)
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            self._encoder = enc
            self.output_dim = enc.config.hidden_size
            logger.info(f"Loaded Wav2Vec2: {model_name} (dim={self.output_dim})")
        except Exception as e:
            logger.warning(f"Wav2Vec2 unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)
        out = self._encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state.float()
        if attention_mask is not None:
            feat_lengths = self._encoder._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_time = hs.size(1)
            feat_mask = torch.arange(max_time, device=hs.device).unsqueeze(0) < feat_lengths.unsqueeze(1)
            feat_mask = feat_mask.unsqueeze(-1).float()
            pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class WavLMEncoder(nn.Module):
    """WavLM, falls back to spectrogram."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze: bool = True):
        super().__init__()
        local_path = os.environ.get("CONFLICTNET_WAVLM_PATH")
        if local_path:
            model_name = local_path
            logger.info(f"[WavLM] Using local path: {local_path}")
        self._encoder = None
        self.output_dim = 768
        try:
            from transformers import WavLMModel
            enc = WavLMModel.from_pretrained(model_name)
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            self._encoder = enc
            self.output_dim = enc.config.hidden_size
            logger.info(f"Loaded WavLM: {model_name} (dim={self.output_dim})")
        except Exception as e:
            logger.warning(f"WavLM unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)
        out = self._encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state.float()
        if attention_mask is not None:
            feat_lengths = self._encoder._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_time = hs.size(1)
            feat_mask = torch.arange(max_time, device=hs.device).unsqueeze(0) < feat_lengths.unsqueeze(1)
            feat_mask = feat_mask.unsqueeze(-1).float()
            pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class Emotion2VecEncoder(nn.Module):
    """Emotion2Vec — tries funasr, then WavLM, then spectrogram fallback."""

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = 768
        self._freeze = freeze
        self._model = WavLMEncoder(freeze=freeze)
        self._backend = "fallback_wavlm"
        self._try_funasr()

    def _try_funasr(self):
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model=self.model_name,
                disable_update=True,
                disable_pipeline=True,
            )
            self._backend = "funasr"
            logger.info(f"[Emotion2Vec] funasr backend: {self.model_name}")
        except Exception as e:
            logger.warning(f"[Emotion2Vec] funasr unavailable ({e}), using WavLM/spectrogram")

    def forward(self, audio, attention_mask=None, return_frames=False):
        if self._backend == "funasr":
            pooled = self._forward_funasr(audio)
            if return_frames:
                return pooled, None
            return pooled
        return self._model(audio, attention_mask=attention_mask, return_frames=return_frames)

    def _forward_funasr(self, audio):
        audio_np = audio.cpu().numpy()
        results = []
        for i in range(audio_np.shape[0]):
            emb = self._model.generate(input=audio_np[i], output_dir="./tmp_funasr")
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict):
                emb = emb[0].get("feats", emb[0])
            emb = np.mean(emb, axis=0) if emb.ndim > 1 else emb
            results.append(torch.from_numpy(emb).float())
        return torch.stack(results).to(audio.device)

    @property
    def device(self) -> torch.device:
        if self._backend == "funasr" and hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device


def build_audio_encoder(name: str = "emotion2vec", **kwargs) -> nn.Module:
    encoders = {
        "wav2vec2": Wav2Vec2Encoder,
        "wavlm": WavLMEncoder,
        "emotion2vec": Emotion2VecEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown audio encoder: {name}. Choose from {list(encoders.keys())}")
    return encoders[name](**kwargs)