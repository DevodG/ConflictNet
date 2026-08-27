#!/usr/bin/env python3
"""Compute per-speaker prosody statistics across all datasets.

Statistics are computed from TRAINING data only to prevent data leakage.
Z-scores for all utterances (train+val+test) use training-only statistics.

Usage:
    python scripts/compute_prosody_stats.py \
        --iemocap_root /data/iemocap \
        --mustard_root /data/mustard \
        --output_file prosody_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Scanner return type: (path, speaker_id, gender, is_train)
ScanItem = Tuple[str, str, Optional[str], bool]


def extract_prosody_single(path: str) -> Optional[Dict[str, Any]]:
    """Extract prosody features from a single audio file.

    Returns dict with f0_mean, energy_mean, speaking_rate or None on failure.
    """
    try:
        import parselmouth
        import librosa
        audio_np, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        return None

    try:
        snd = parselmouth.Sound(audio_np, sampling_frequency=sr)
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0

        intensity = snd.to_intensity()
        intensities = intensity.values.T.squeeze()
        energy_mean = float(np.mean(intensities))

        duration = len(audio_np) / sr
        speaking_rate = len(f0_voiced) / max(duration, 1e-6)

        return {
            "f0_mean": f0_mean,
            "energy_mean": energy_mean,
            "speaking_rate": speaking_rate,
        }
    except Exception:
        # Fallback to librosa
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio_np.astype(np.float32),
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
            f0_voiced = f0[voiced_flag]
            f0_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) else 0.0
            rms = librosa.feature.rms(y=audio_np.astype(np.float32))[0]
            energy_mean = float(np.mean(rms))
            duration = len(audio_np) / sr
            speaking_rate = len(f0_voiced) / max(duration, 1e-6)
            return {
                "f0_mean": f0_mean,
                "energy_mean": energy_mean,
                "speaking_rate": speaking_rate,
            }
        except Exception:
            return None


def scan_iemocap(
    root: str,
    train_sessions: Tuple[int, ...] = (1, 2, 3, 4),
) -> List[ScanItem]:
    """Scan IEMOCAP for (path, speaker_id, gender, is_train).

    Only sessions in ``train_sessions`` are flagged as training data
    for statistics computation. Default: sessions 1-4 are train,
    session 5 is held out (val/test).
    """
    items = []
    root_p = Path(root)
    for sess_dir in sorted(root_p.glob("Session*")):
        sess_name = sess_dir.name
        sess_num = int(sess_name.replace("Session", ""))
        is_train = sess_num in train_sessions
        wav_root = sess_dir / "sentences" / "wav"
        if not wav_root.exists():
            continue
        for wav in wav_root.rglob("*.wav"):
            utt_id = wav.stem
            speaker = utt_id[:6]  # e.g. 'Ses01F' — session+gender uniquely identifies speaker
            gender = "F" if len(utt_id) > 5 and utt_id[5] == "F" else "M"
            items.append((str(wav), speaker, gender, is_train))
    return items


def scan_mustard(
    root: str,
    json_path: Optional[str] = None,
) -> List[ScanItem]:
    """Scan MUStARD for (path, speaker_id, gender, is_train).

    Uses speaker-stratified split: group by speaker, assign whole speakers to train/val.
    """
    # Load data from JSON if available
    data = {}
    if json_path and Path(json_path).exists():
        with open(json_path) as f:
            data = json.load(f)

    items = []
    root_p = Path(root)
    wav_dirs = list(root_p.rglob("*.wav"))
    
    # Collect all samples with speaker info
    all_samples = []
    for wav in wav_dirs:
        spk = wav.parent.stem
        # Find matching key in JSON data
        matching_key = None
        for key in data.keys():
            if key in wav.stem:
                matching_key = key
                break
        sarcasm = int(data.get(matching_key, {}).get("sarcasm", 0)) if matching_key else 0
        all_samples.append({
            "wav_path": str(wav),
            "speaker_id": f"mustard_{spk}",
            "sarcasm": sarcasm,
        })

    # Speaker-stratified split: group by speaker, assign whole speakers to train/val
    speaker_groups: Dict[str, List[Dict]] = {}
    for s in all_samples:
        speaker_groups.setdefault(s["speaker_id"], []).append(s)
    rng = random.Random(42)
    speaker_ids = list(speaker_groups.keys())
    rng.shuffle(speaker_ids)

    train_items: List[Dict] = []
    val_items: List[Dict] = []
    target_train = int(len(all_samples) * 0.8)
    for sid in speaker_ids:
        samples = speaker_groups[sid]
        if not train_items or len(train_items) + len(samples) <= target_train:
            train_items.extend(samples)
        else:
            val_items.extend(samples)

    train_speakers = {s["speaker_id"] for s in train_items}
    
    for sample in all_samples:
        is_train = sample["speaker_id"] in train_speakers
        items.append((sample["wav_path"], sample["speaker_id"], None, is_train))
    
    return items


def scan_cremad(
    root: str,
    train_ratio: float = 0.8,
) -> List[ScanItem]:
    """Scan CREMA-D for (path, speaker_id, gender, is_train).

    Uses speaker-stratified split: group by actor, assign whole actors to train/val.
    """
    items = []
    wav_dir = Path(root) / "AudioWAV"
    if not wav_dir.exists():
        return items
    all_wavs = list(wav_dir.glob("*.wav"))

    # Collect all samples with actor info
    all_samples = []
    for wav in all_wavs:
        parts = wav.stem.split("_")
        if len(parts) >= 1:
            actor_id = parts[0]
            speaker = f"cremad_{actor_id}"
            all_samples.append({
                "wav_path": str(wav),
                "speaker_id": speaker,
            })

    # Speaker-stratified split: group by actor, assign whole actors to train/val
    speaker_groups: Dict[str, List[Dict]] = {}
    for s in all_samples:
        speaker_groups.setdefault(s["speaker_id"], []).append(s)
    rng = random.Random(42)
    speaker_ids = list(speaker_groups.keys())
    rng.shuffle(speaker_ids)

    train_items: List[Dict] = []
    val_items: List[Dict] = []
    target_train = int(len(all_samples) * train_ratio)
    for sid in speaker_ids:
        samples = speaker_groups[sid]
        if not train_items or len(train_items) + len(samples) <= target_train:
            train_items.extend(samples)
        else:
            val_items.extend(samples)

    train_speakers = {s["speaker_id"] for s in train_items}

    for sample in all_samples:
        is_train = sample["speaker_id"] in train_speakers
        items.append((sample["wav_path"], sample["speaker_id"], None, is_train))
    return items


def scan_meld(
    root: str,
    train_splits: Tuple[str, ...] = ("train",),
) -> List[ScanItem]:
    """Scan MELD for (path, speaker_id, gender, is_train).

    Only splits in ``train_splits`` are flagged as training data.
    Default: only "train" split contributes to statistics.
    """
    items = []
    root_p = Path(root)
    for split_name in ("train", "dev", "test"):
        wav_dir = root_p / split_name / f"{split_name}_splits"
        if not wav_dir.exists():
            continue
        is_train = split_name in train_splits
        for wav in wav_dir.glob("*.wav"):
            # Use dialogue prefix as speaker proxy (no per-speaker dirs in MELD)
            speaker = f"meld_{split_name}"
            items.append((str(wav), speaker, None, is_train))
    return items


class WelfordAggregator:
    """Online mean/std tracking using Welford's algorithm."""

    def __init__(self, dim: int = 3):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(len(self.mean), dtype=np.float64)
        return np.sqrt(self.M2 / (self.n - 1))


def parse_args():
    p = argparse.ArgumentParser(description="Compute per-speaker prosody statistics (training-data only)")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--iemocap_train_sessions", type=str, default="1,2,3,4",
                   help="Comma-separated session numbers for training statistics (default: 1,2,3,4)")
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--mustard_json", type=str, default=None,
                   help="Path to MUStARD++ JSON file for train/val split detection")
    p.add_argument("--cremad_root", type=str, default=None)
    p.add_argument("--cremad_train_ratio", type=float, default=0.8,
                   help="CREMA-D training ratio (default: 0.8)")
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--meld_train_splits", type=str, default="train",
                   help="Comma-separated MELD splits for training (default: train)")
    p.add_argument("--output_file", type=str, default="prosody_stats.json")
    p.add_argument("--max_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    all_items: List[ScanItem] = []
    if args.iemocap_root:
        train_sessions = tuple(int(s.strip()) for s in args.iemocap_train_sessions.split(","))
        all_items.extend(scan_iemocap(args.iemocap_root, train_sessions=train_sessions))
    if args.mustard_root:
        all_items.extend(scan_mustard(args.mustard_root, json_path=args.mustard_json))
    if args.cremad_root:
        all_items.extend(scan_cremad(args.cremad_root, train_ratio=args.cremad_train_ratio))
    if args.meld_root:
        train_splits = tuple(s.strip() for s in args.meld_train_splits.split(","))
        all_items.extend(scan_meld(args.meld_root, train_splits=train_splits))

    if not all_items:
        logger.error("No audio files found. Provide at least one dataset root.")
        sys.exit(1)

    logger.info(f"Scanned {len(all_items)} audio files")

    # Extract prosody in parallel
    paths_only = [item[0] for item in all_items]
    prosody_results: List[Optional[Dict[str, Any]]] = [None] * len(paths_only)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        fut_map = {executor.submit(extract_prosody_single, p): i for i, p in enumerate(paths_only)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                prosody_results[idx] = fut.result()
            except Exception:
                prosody_results[idx] = None

    # Aggregate per speaker using TRAINING DATA ONLY (prevents L1 data leakage)
    speaker_data: Dict[str, Dict[str, Any]] = {}
    global_agg = WelfordAggregator(dim=3)
    n_train = 0
    n_val = 0
    for (path, spk_id, gender, is_train), result in zip(all_items, prosody_results):
        if result is None:
            continue
        if is_train:
            n_train += 1
            global_agg.update(np.array([result["f0_mean"], result["energy_mean"], result["speaking_rate"]]))
            if spk_id not in speaker_data:
                speaker_data[spk_id] = {
                    "agg": WelfordAggregator(dim=3),
                    "n": 0,
                    "genders": {},
                }
            sd = speaker_data[spk_id]
            sd["agg"].update(np.array([result["f0_mean"], result["energy_mean"], result["speaking_rate"]]))
            sd["n"] += 1
            if gender:
                sd["genders"][gender] = sd["genders"].get(gender, 0) + 1
        else:
            n_val += 1

    logger.info(f"Using {n_train} training utterances for statistics ({n_val} val/test excluded from stats)")

    # Serialize per-speaker stats (training-only)
    output: Dict[str, Dict[str, Any]] = {}
    for spk_id, sd in speaker_data.items():
        output[spk_id] = {
            "mean": sd["agg"].mean.tolist(),
            "std": sd["agg"].std.tolist(),
            "n": sd["n"],
            "genders": sd["genders"],
        }
    # Held-out speakers must not receive zero vectors: use a population prior
    # fitted only on training audio. This preserves split isolation while
    # retaining meaningful scale for speaker-disjoint evaluation.
    output["__global__"] = {
        "mean": global_agg.mean.tolist(),
        "std": global_agg.std.tolist(),
        "n": global_agg.n,
        "genders": {},
    }

    stats_path = Path(args.output_file)
    with open(stats_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Computed prosody stats for {len(output)} speakers → {stats_path}")

    # Compute per-utterance z-scores SEPARATELY for train and val using TRAINING-ONLY statistics
    # This fixes L2: train and val collate functions receive DIFFERENT z-score files
    train_zscores: Dict[str, List[float]] = {}
    val_zscores: Dict[str, List[float]] = {}
    
    for (path, spk_id, _gender, is_train), result in zip(all_items, prosody_results):
        if result is None:
            continue
        stats = output.get(spk_id, output["__global__"])
        spk_mean = np.array(stats["mean"], dtype=np.float32)
        spk_std = np.array(stats["std"], dtype=np.float32)
        spk_std = np.clip(spk_std, 1e-6, None)
        feat = np.array([result["f0_mean"], result["energy_mean"], result["speaking_rate"]], dtype=np.float32)
        z = (feat - spk_mean) / spk_std
        utt_id = Path(path).stem
        z_list = z.tolist()
        if is_train:
            train_zscores[utt_id] = z_list
        else:
            val_zscores[utt_id] = z_list

    train_zscores_path = stats_path.with_suffix(".train.zscores.json")
    val_zscores_path = stats_path.with_suffix(".val.zscores.json")
    with open(train_zscores_path, "w") as f:
        json.dump(train_zscores, f)
    with open(val_zscores_path, "w") as f:
        json.dump(val_zscores, f)
    
    logger.info(f"Computed per-utterance TRAIN z-scores for {len(train_zscores)} utterances → {train_zscores_path}")
    logger.info(f"Computed per-utterance VAL z-scores for {len(val_zscores)} utterances → {val_zscores_path}")
    logger.info("Note: z-scores use training-data-only speaker statistics (no data leakage)")


if __name__ == "__main__":
    main()
