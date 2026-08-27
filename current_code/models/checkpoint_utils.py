"""Safe checkpoint loading utilities.

Centralises all checkpoint I/O so that:
  1. ``.safetensors`` files are preferred (pickle-free, CWE-502 safe).
  2. Legacy ``.pt`` / ``.pth`` files are loaded with ``weights_only=True``
     to prevent arbitrary code execution via pickle.
  3. A single call-site makes auditing straightforward.

Usage::

    from models.checkpoint_utils import load_checkpoint_state

    state = load_checkpoint_state("checkpoints/best_model.pt", device="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Union

import torch

logger = logging.getLogger(__name__)


def load_checkpoint_state(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",  # type: ignore[arg-type]
) -> Dict[str, Any]:
    """Load checkpoint weights from *path* safely.

    Supports two formats:

    * ``.safetensors`` — loaded via ``safetensors.torch.load_file``
      (no pickle, no arbitrary code execution).
    * ``.pt`` / ``.pth`` / other — loaded via ``torch.load`` with
      ``weights_only=True`` (restricts unpickling to tensor data only,
      blocking arbitrary object instantiation).

    For legacy ``.pt`` files the returned dict is the raw checkpoint.
    For ``.safetensors`` the returned dict is the flat state-dict (no
    ``model_state_dict`` wrapper), equivalent to
    ``torch.load(...)["model_state_dict"]``.

    Args:
        path: Filesystem path to the checkpoint file.
        device: Target device (``"cpu"`` / ``"cuda"`` / ``torch.device``).

    Returns:
        A dictionary of tensors (state-dict or full checkpoint dict).

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be deserialised safely.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_str = str(device)

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as st_load

        logger.debug("Loading safetensors checkpoint: %s", checkpoint_path)
        return st_load(str(checkpoint_path), device=device_str)

    # Legacy .pt / .pth — weights_only=True prevents arbitrary pickle execution
    logger.debug("Loading legacy .pt checkpoint: %s (weights_only=True)", checkpoint_path)
    return torch.load(  # noqa: S301 — weights_only=True prevents arbitrary code execution
        str(checkpoint_path),
        map_location=device_str,
        weights_only=True,
    )


def extract_model_state(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the model state-dict from a checkpoint.

    Handles both flat state-dicts (e.g. from safetensors) and wrapped
    checkpoints containing a ``model_state_dict`` key.
    """
    return checkpoint.get("model_state_dict", checkpoint)


def load_conflictnet_model(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
):
    """Reconstruct ConflictNet from checkpoint metadata and load it safely.

    Rebuilding with constructor defaults is a silent and serious source of
    invalid evaluation (audio encoder, LoRA, ablation toggles, and embedding
    size can all differ from training). The trainer writes these values to the
    sibling ``*_meta.json`` file, so every inference entry point should use
    this helper.
    """
    from models.conflictnet import ConflictNet

    checkpoint_path = Path(path)
    state = extract_model_state(load_checkpoint_state(checkpoint_path, device=device))
    meta_path = checkpoint_path.parent / f"{checkpoint_path.stem}_meta.json"
    config: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            config = json.load(f).get("experiment_config", {})
    else:
        logger.warning("No checkpoint metadata at %s; using constructor defaults", meta_path)

    model = ConflictNet(
        audio_encoder_name=config.get("audio_encoder", "emotion2vec"),
        embed_dim=int(config.get("embed_dim", 256)),
        lora_r=int(config.get("lora_r", 16)),
        use_speaker_norm=bool(config.get("use_speaker_norm", True)),
        use_temporal=bool(config.get("use_temporal", True)),
        use_cross_attn_injection=bool(config.get("use_cross_attn_injection", True)),
        use_speaker_adaptive_threshold=bool(config.get("use_speaker_adaptive_threshold", True)),
        use_baseline_subtract=bool(config.get("use_baseline_subtract", True)),
        use_word_divergence=bool(config.get("use_word_divergence", True)),
        temporal_max_turns=int(config.get("temporal_max_turns", 16)),
        focal_loss_gamma=float(config.get("focal_loss_gamma", 0.0)),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        separation_lambda=float(config.get("separation_lambda", 0.1)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval(), config
