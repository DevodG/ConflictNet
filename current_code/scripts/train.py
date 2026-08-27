"""CLI: train ConflictNet v2.

Usage:
    python scripts/train.py --config configs/default.yaml \
        --iemocap_root /data/iemocap \
        --mustard_root  /data/mustard \
        --output_dir checkpoints/run1 \
        --pretrain_epochs 5 \
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Fix for macOS OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to sys.path so 'data', 'models' etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ConflictNet v2")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--cremad_root", type=str, default=None, help="CREMA-D dataset root")
    p.add_argument("--meld_root", type=str, default=None, help="MELD dataset root")
    p.add_argument("--musan_path", type=str, default=None, help="MUSAN corpus for noise augmentation")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--audio_encoder", type=str, default="emotion2vec",
                   choices=["emotion2vec", "wavlm", "wav2vec2"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_speaker_norm", action="store_true")
    p.add_argument("--no_temporal", action="store_true",
                   help="Disable Transformer temporal context module")
    p.add_argument("--no_cross_attn_injection", action="store_true",
                   help="Disable cross-attention injection from temporal context into audio+text")
    p.add_argument("--no_speaker_adaptive_threshold", action="store_true",
                   help="Disable speaker-adaptive divergence threshold (use fixed threshold)")
    p.add_argument("--no_baseline_subtract", action="store_true",
                   help="Disable baseline-subtract prosody normalisation (use z-score instead)")
    p.add_argument("--no_word_divergence", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes (auto-reduced to 2 on MPS)")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Samples prefetched per worker (default 2)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--prosody_stats", type=str, default=None,
                   help="Path to .pt file from compute_prosody_stats.py with per-utterance z-scores")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision (fp16) training")
    p.add_argument("--focal_loss_gamma", type=float, default=0.0,
                   help="Focal loss gamma (0 = standard BCE, 2 = recommended)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing epsilon for multi-label BCE (0 = none, 0.1 = mild)")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Path to local tokenizer directory (avoids HuggingFace download)")
    p.add_argument("--target_f1", type=float, default=0.0,
                   help="Target val F1. If not met after training, resume with halved LR (0 = disable)")
    p.add_argument("--max_retries", type=int, default=0,
                   help="Max times to continue training if below target_f1")
    p.add_argument("--resume_epochs", type=int, default=10,
                   help="Additional epochs per retry")
    p.add_argument("--dry_run", action="store_true",
                   help="Run 1 batch through validation, report shapes/memory, then exit")
    return p.parse_args()


def main():
    args = parse_args(argv=None)

    from models.device_utils import resolve_device, mps_optimized_config

    if args.device is None:
        args.device = resolve_device()

    # Apply M2-optimized defaults when running on Apple Silicon
    if args.device == "mps":
        overrides = mps_optimized_config()
        for key, val in overrides.items():
            if hasattr(args, key) and getattr(args, key) == argparse.SUPPRESS:
                continue
            if key == "batch_size" and args.batch_size != 16:
                continue
            if key == "num_workers" and args.num_workers != 4:
                continue
            if key in ("use_temporal", "use_cross_attn_injection",
                       "use_word_divergence", "use_speaker_adaptive_threshold"):
                continue
            setattr(args, key, val)

        # Auto-disable memory-heavy features not useful for single-utterance datasets
        if args.no_word_divergence is False:
            args.no_word_divergence = True
            logger.info("[MPS] Disabled word_divergence (requires MFA alignment)")
        if args.no_temporal is False:
            args.no_temporal = True
            logger.info("[MPS] Disabled temporal context (single-utterance dataset)")
        if args.no_cross_attn_injection is False:
            args.no_cross_attn_injection = True
            logger.info("[MPS] Disabled cross-attn injection (no temporal context)")
        if args.no_speaker_adaptive_threshold is False:
            args.no_speaker_adaptive_threshold = True
            logger.info("[MPS] Disabled speaker-adaptive threshold (reduces params)")

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    elif args.device == "mps":
        torch.mps.manual_seed(args.seed)

    # --- Build model ---
    from models.conflictnet import ConflictNet

    model = ConflictNet(
        audio_encoder_name=args.audio_encoder,
        embed_dim=args.embed_dim,
        use_speaker_norm=not args.no_speaker_norm,
        use_temporal=not args.no_temporal,
        use_cross_attn_injection=not args.no_cross_attn_injection,
        use_speaker_adaptive_threshold=not args.no_speaker_adaptive_threshold,
        use_baseline_subtract=not args.no_baseline_subtract,
        use_word_divergence=not args.no_word_divergence,
        lora_r=args.lora_r,
        focal_loss_gamma=args.focal_loss_gamma,
        label_smoothing=args.label_smoothing,
    )

    param_counts = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in param_counts.values())
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # --- Dry-run validation (uses synthetic data, no dataset roots needed) ---
    if args.dry_run:
        logger.info("=" * 50)
        logger.info("DRY RUN: validating pipeline with synthetic data")
        model.to(args.device)
        model.eval()
        B = 2
        audio = torch.randn(B, 48000, device=args.device)
        input_ids = torch.randint(0, 1000, (B, 128), device=args.device)
        attention_mask = torch.ones(B, 128, dtype=torch.long, device=args.device)
        audio_attention_mask = torch.ones(B, 48000, dtype=torch.bool, device=args.device)
        with torch.no_grad(), torch.autocast(device_type=args.device, enabled=args.amp):
            out = model(
                audio=audio, input_ids=input_ids, attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
                conflict_type_labels=torch.randint(0, 2, (B, 3), device=args.device).float(),
                severity_labels=torch.rand(B, 1, device=args.device),
                conflict_binary_labels=torch.randint(0, 2, (B,), device=args.device).float(),
                pretraining=True,
            )
        logger.info(f"  audio:          {tuple(audio.shape)}")
        logger.info(f"  input_ids:      {tuple(input_ids.shape)}")
        logger.info(f"  logits_type:    {tuple(out.logits_type.shape)}")
        logger.info(f"  probs_type:     {tuple(out.probs_type.shape)}")
        logger.info(f"  severity:       {out.severity.shape if out.severity is not None else 'N/A'}")
        logger.info(f"  conflict_flag:  {out.conflict_flag.shape}")
        logger.info(f"  audio_embed:    {tuple(out.audio_embed.shape)}")
        logger.info(f"  text_embed:     {tuple(out.text_embed.shape)}")
        logger.info(f"  fused_embed:    {tuple(out.fused_embed.shape)}")
        logger.info(f"  context_pooled: {tuple(out.context_pooled.shape)}")
        logger.info(f"  loss:           {out.loss.item():.4f}")
        logger.info(f"  breakdown:      {out.loss_breakdown}")
        logger.info(f"  params:         {total_trainable:,} trainable / "
                     f"{sum(v['total'] for v in param_counts.values()):,} total")
        if args.device == "mps":
            logger.info(f"  MPS allocated:  {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
        logger.info("Dry run complete — exiting.")
        sys.exit(0)

    # --- Build datasets ---
    from data.datasets import (
        IEMOCAPDataset, MUStARDDataset, CREMADDataset, MELDDataset,
        make_collate_fn,
    )

    from data.augmentation import AudioAugmentor

    prosody_lookup = None
    if args.prosody_stats:
        prosody_path = Path(args.prosody_stats)
        zscores_p = prosody_path.parent / f"{prosody_path.stem}.zscores.json"
        if not zscores_p.exists():
            zscores_p = prosody_path.with_suffix(".zscores.json")
        if zscores_p.exists():
            try:
                with open(zscores_p) as f:
                    raw = json.load(f)
                prosody_lookup = {k: torch.tensor(v) for k, v in raw.items()}
                if not prosody_lookup:
                    prosody_lookup = None
                else:
                    logger.info(f"[Prosody] loaded {len(prosody_lookup)} z-score entries from {zscores_p}")
            except Exception as e:
                logger.warning(f"[Prosody] failed to load {zscores_p}: {e}")
        else:
            logger.warning(f"[Prosody] lookup file not found: {zscores_p}")

    augmentor = AudioAugmentor(
        sample_rate=16000,
        musan_path=getattr(args, "musan_path", None),
    )
    train_collate = make_collate_fn(augmentor=augmentor, prosody_lookup=prosody_lookup)
    val_collate = make_collate_fn(prosody_lookup=prosody_lookup)

    train_datasets = []
    val_datasets = []

    if args.iemocap_root:
        train_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[1, 2, 3, 4]))
        val_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[5]))

    if args.mustard_root:
        train_datasets.append(MUStARDDataset(args.mustard_root, split="train"))
        val_datasets.append(MUStARDDataset(args.mustard_root, split="val"))

    if args.cremad_root:
        tok_kwargs = {"tokenizer_name": args.tokenizer_path} if args.tokenizer_path else {}
        train_datasets.append(CREMADDataset(args.cremad_root, split="train", **tok_kwargs))
        val_datasets.append(CREMADDataset(args.cremad_root, split="val", **tok_kwargs))

    if args.meld_root:
        train_datasets.append(MELDDataset(args.meld_root, split="train"))
        val_datasets.append(MELDDataset(args.meld_root, split="val"))

    if not train_datasets:
        raise ValueError("Provide at least one of --iemocap_root, --mustard_root, --cremad_root, or --meld_root")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    logger.info(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    from models.device_utils import supports_pin_memory

    pin = supports_pin_memory(args.device)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=pin,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    loader_kwargs["collate_fn"] = val_collate
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # --- Trainer ---
    from training.trainer import ConflictNetTrainer, get_warmup_cosine_scheduler
    from models.experiment_config import ExperimentConfig

    exp_config = ExperimentConfig.from_args(args)
    cfg = exp_config.to_dict()
    trainer = ConflictNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        exp_config=exp_config,
        device=args.device,
        output_dir=args.output_dir,
    )

    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    retries = 0
    while True:
        trainer.train(n_epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, start_epoch=start_epoch)

        if args.max_retries <= 0 or args.target_f1 <= 0:
            break

        meta_path = Path(args.output_dir) / "best_model_meta.json"
        best_f1 = 0.0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best_f1 = meta.get("best_val_f1", 0.0)

        logger.info(f"[Retry {retries+1}/{args.max_retries}] Best val F1 = {best_f1:.4f}, target = {args.target_f1}")

        if best_f1 >= args.target_f1:
            logger.info(f"Target F1 {args.target_f1} reached!")
            break

        if retries >= args.max_retries:
            logger.info(f"Max retries ({args.max_retries}) exhausted, best F1 = {best_f1:.4f}")
            break

        retries += 1
        args.lr = float(args.lr) / 2
        args.resume_from = str(Path(args.output_dir) / "best_model.safetensors")
        args.pretrain_epochs = 0

        start_epoch = trainer.load_checkpoint(args.resume_from)
        args.epochs = start_epoch + args.resume_epochs

        logger.info(f"Resuming (retry {retries}/{args.max_retries}, lr={args.lr:.2e}, epochs {start_epoch}–{args.epochs-1})")

        # Reset scheduler for continuation — cosine decays new_lr → 0 over resume_epochs
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr
        steps_per_epoch = len(trainer.train_loader)
        trainer.scheduler = get_warmup_cosine_scheduler(
            trainer.optimizer,
            num_warmup_steps=0,
            num_training_steps=steps_per_epoch * args.resume_epochs,
        )


if __name__ == "__main__":
    main()
