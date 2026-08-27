#!/usr/bin/env python3
"""Ensemble multiple ConflictNet checkpoints by averaging predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble ConflictNet checkpoints")
    p.add_argument("--checkpoints", type=str, required=True,
                   help="Comma-separated list of checkpoint paths")
    p.add_argument("--cache_path", type=str, required=True,
                   help="Path to pre-computed embeddings .pt file")
    p.add_argument("--output_dir", type=str, default="checkpoints/ensemble",
                   help="Output directory for ensemble predictions")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    from models.cached_conflictnet import CachedConflictNet
    
    model = CachedConflictNet(
        embed_dim=512,
        use_speaker_norm=True,
        use_temporal=True,
        use_cross_attn_injection=True,
        use_speaker_adaptive_threshold=True,
        use_word_divergence=False,
        use_swap_pretraining=True,
        focal_loss_gamma=2.0,
        label_smoothing=0.1,
        separation_lambda=0.1,
    )
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    
    from models.device_utils import resolve_device
    device = args.device or resolve_device()
    
    checkpoint_paths = [p.strip() for p in args.checkpoints.split(",")]
    logger.info(f"Loading {len(checkpoint_paths)} checkpoints...")
    
    models = []
    for cp in checkpoint_paths:
        logger.info(f"  Loading {cp}")
        model = load_checkpoint(cp, device)
        models.append(model)
    
    # Load validation dataset
    from data.cached_dataset import CachedEmbeddingDataset, cached_collate_fn
    val_set = CachedEmbeddingDataset(args.cache_path, split="val")
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=cached_collate_fn,
        pin_memory=False,
    )
    
    logger.info(f"Val samples: {len(val_set)}")
    
    # Ensemble inference
    all_probs_type = []
    all_severity = []
    all_conflict_flag = []
    all_labels_type = []
    all_labels_severity = []
    all_labels_binary = []
    all_has_real_type = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            batch_probs_type = []
            batch_severity = []
            batch_conflict_flag = []
            
            for model in models:
                out = model(
                    audio_embed=batch["audio_embed"],
                    text_embed=batch["text_embed"],
                    speaker_feat=batch["speaker_feat"],
                    conflict_type_labels=batch.get("conflict_type_labels"),
                    severity_labels=batch.get("severity"),
                    conflict_binary_labels=batch.get("conflict_binary"),
                    has_real_type_labels=batch.get("has_real_type_labels"),
                )
                batch_probs_type.append(out.probs_type)
                if out.severity is not None:
                    batch_severity.append(out.severity)
                batch_conflict_flag.append(out.conflict_flag.float())
            
            # Average predictions
            avg_probs_type = torch.stack(batch_probs_type).mean(0)
            avg_severity = torch.stack(batch_severity).mean(0) if batch_severity else None
            avg_conflict_flag = (torch.stack(batch_conflict_flag).mean(0) > 0.5).float()
            
            all_probs_type.append(avg_probs_type.cpu())
            if avg_severity is not None:
                all_severity.append(avg_severity.cpu())
            all_conflict_flag.append(avg_conflict_flag.cpu())
            all_labels_type.append(batch.get("conflict_type_labels", torch.empty(0)).cpu())
            all_labels_severity.append(batch.get("severity", torch.empty(0)).cpu())
            all_labels_binary.append(batch.get("conflict_binary", torch.empty(0)).cpu())
            all_has_real_type.append(batch.get("has_real_type_labels", torch.empty(0)).cpu())
    
    # Concatenate
    all_probs_type = torch.cat(all_probs_type, 0)
    all_conflict_flag = torch.cat(all_conflict_flag, 0)
    all_labels_type = torch.cat(all_labels_type, 0) if len(all_labels_type) > 0 else None
    all_labels_binary = torch.cat(all_labels_binary, 0) if len(all_labels_binary) > 0 else None
    all_has_real_type = torch.cat(all_has_real_type, 0) if len(all_has_real_type) > 0 else None
    
    if all_severity:
        all_severity = torch.cat(all_severity, 0)
    
    # Compute metrics
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    
    # Binary conflict F1
    binary_f1 = f1_score(all_labels_binary.numpy(), all_conflict_flag.numpy())
    logger.info(f"Ensemble Binary Conflict F1: {binary_f1:.4f}")
    
    # Multi-label type F1 (only on real labels)
    if all_labels_type is not None and all_has_real_type is not None:
        real_mask = all_has_real_type.bool()
        if real_mask.any():
            preds_type = (all_probs_type[real_mask] > 0.5).float()
            labels_type = all_labels_type[real_mask]
            type_f1_micro = f1_score(labels_type.numpy(), preds_type.numpy(), average='micro')
            type_f1_macro = f1_score(labels_type.numpy(), preds_type.numpy(), average='macro')
            logger.info(f"Ensemble Type F1 (micro): {type_f1_micro:.4f}")
            logger.info(f"Ensemble Type F1 (macro): {type_f1_macro:.4f}")
            
            # Per-class
            for i, name in enumerate(["sarcasm", "suppression", "deception"]):
                if labels_type[:, i].sum() > 0:
                    f1 = f1_score(labels_type[:, i].numpy(), preds_type[:, i].numpy())
                    logger.info(f"  {name}: F1={f1:.4f}")
    
    # Severity MSE
    if all_severity is not None and all_labels_severity is not None:
        valid = ~torch.isnan(all_labels_severity)
        if valid.any():
            mse = torch.nn.functional.mse_loss(all_severity[valid], all_labels_severity[valid])
            logger.info(f"Ensemble Severity MSE: {mse.item():.4f}")
    
    # Save ensemble predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "probs_type": all_probs_type,
        "conflict_flag": all_conflict_flag,
        "severity": all_severity if all_severity else None,
        "labels_type": all_labels_type,
        "labels_severity": all_labels_severity,
        "labels_binary": all_labels_binary,
    }, output_dir / "ensemble_predictions.pt")
    
    logger.info(f"Saved ensemble predictions to {output_dir / 'ensemble_predictions.pt'}")


if __name__ == "__main__":
    main()