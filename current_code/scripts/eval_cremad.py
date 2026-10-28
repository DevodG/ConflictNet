#!/usr/bin/env python3
"""Evaluate a trained ConflictNet checkpoint on the CREMA-D validation split.

CREMA-D has no real conflict-subtype annotations, so this script reports
BINARY conflict metrics only (``conflict_flag`` vs ``conflict_binary``).
Subtype macro-F1/AP/AUC on CREMA-D would be pure emotion-proxy artifacts
and are therefore refused — see ``scripts/evaluate.py`` for subtype eval
on datasets with real labels (MUStARD++/CASE).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate ConflictNet on CREMA-D val (binary only)")
    p.add_argument("--cremad_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="results/cremad")
    p.add_argument("--prosody_stats", type=str, default=None,
                   help="Path to prosody stats JSON (VAL split z-scores are used)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from models.device_utils import resolve_device, supports_pin_memory, supports_non_blocking
    from models.checkpoint_utils import load_conflictnet_model
    from data.datasets import CREMADDataset, load_prosody_lookups, make_collate_fn

    device = args.device or resolve_device()
    model, exp_cfg = load_conflictnet_model(args.checkpoint, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {exp_cfg.get('audio_encoder', '?')}, params: {n_params:,}")

    prosody_lookup = None
    if args.prosody_stats:
        _, prosody_lookup, used_legacy = load_prosody_lookups(args.prosody_stats)
        if used_legacy:
            logger.warning("[CREMA-D] REFUSING legacy combined z-scores for reported metrics")
            prosody_lookup = None

    ds = CREMADDataset(args.cremad_root, split="val")
    if len(ds) == 0:
        raise ValueError(f"No CREMA-D val samples found under {args.cremad_root}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=supports_pin_memory(device), collate_fn=make_collate_fn(prosody_lookup=prosody_lookup),
    )

    all_pred, all_true = [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            bg = {
                k: v.to(device, non_blocking=supports_non_blocking(device))
                if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            out = model(
                audio=bg["audio"], input_ids=bg["input_ids"], attention_mask=bg["attention_mask"],
                audio_attention_mask=bg.get("audio_attention_mask"), prosody_z=bg.get("prosody_z"),
            )
            all_pred.append(out.conflict_flag.cpu().numpy().astype(int))
            all_true.append(b["conflict_binary"].numpy().astype(int))
    elapsed = time.time() - t0

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = {
        "dataset": "cremad",
        "n_samples": len(ds),
        "eval_time_s": round(elapsed, 2),
        "binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "note": "binary-only: CREMA-D has no real subtype labels; subtype macro-F1 refused",
    }
    logger.info(f"[CREMA-D] n={len(ds)} binary_f1={metrics['binary_f1']:.4f} "
                f"acc={metrics['binary_accuracy']:.4f} ({elapsed:.1f}s)")
    with open(out_dir / "cremad_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Report saved to {out_dir}/")


if __name__ == "__main__":
    main()
