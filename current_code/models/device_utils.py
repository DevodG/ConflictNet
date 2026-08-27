from __future__ import annotations

from typing import Dict

import torch


def resolve_device(prefer: str | None = None) -> str:
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_supports_amp(device: str) -> bool:
    return device in ("cuda", "mps")


def supports_pin_memory(device: str) -> bool:
    return device == "cuda"


def supports_non_blocking(device: str) -> bool:
    return device == "cuda"


def mps_optimized_config() -> Dict:
    """Return optimal training defaults when running on Apple Silicon (MPS).

    M2 Macs have unified memory (8–16 GB) and no dedicated VRAM.
    Large frozen encoders (WavLM + DeBERTa = ~620M params) consume
    most of it, leaving little for activations. These defaults trade
    batch size for stability and enable AMP to halve activation memory.

    Returns:
        Dict of overrides suitable for ``**kwargs`` expansion into
        training args or experiment config.
    """
    return {
        "batch_size": 2,
        "num_workers": 0,
        "amp": True,
        "prefetch_factor": 2,
        "gradient_accumulation_steps": 4,
        "temporal_max_turns": 1,
        "use_temporal": False,
        "use_cross_attn_injection": False,
        "use_word_divergence": False,
        "use_speaker_adaptive_threshold": False,
    }
