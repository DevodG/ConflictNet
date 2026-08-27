"""Pre-compute frozen encoder embeddings for fast cached training.

Runs the full ConflictNet ``encode()`` pass (WavLM + DeBERTa + ECAPA)
over the entire dataset once and saves projected embeddings to disk.

Phase 1 of the two-phase training strategy:
  1. (this script)   Pre-compute audio_embed, text_embed, speaker_feat
  2. (train.py --use_cached)  Train fusion → classifier on cached embeddings

Usage:
    python scripts/precompute_embeddings.py \\
        --cremad_root data/cremad \\
        --audio_encoder wavlm \\
        --output_dir cached_embeddings
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pre-compute encoder embeddings")
    p.add_argument("--cremad_root", type=str, default=None)
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--iemocap_train_sessions", type=str, default="1,2,3,4",
                   help="Comma-separated session numbers for training (default: 1,2,3,4)")
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--audio_encoder", type=str, default="wavlm",
                   choices=["emotion2vec", "wavlm", "wav2vec2"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for CPU decoding/tokenization")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="cached_embeddings")
    p.add_argument("--prosody_stats", type=str, default=None)
    p.add_argument("--tokenizer_path", type=str, default=None)
    return p.parse_args(argv)


@torch.no_grad()
def main():
    args = parse_args()

    from models.device_utils import resolve_device, mps_optimized_config, supports_pin_memory, supports_non_blocking

    if args.device is None:
        args.device = resolve_device()

    mps_cfg = mps_optimized_config()
    batch_size = min(args.batch_size, mps_cfg["batch_size"]) if args.device == "mps" else args.batch_size

    # Build datasets
    from data.datasets import (
        IEMOCAPDataset, MUStARDDataset, CREMADDataset, MELDDataset,
        make_collate_fn,
    )

    # No augmentation during pre-compute (augmentation applied per-epoch, not here)
    collate_fn = make_collate_fn()

    datasets = []
    if args.cremad_root:
        tok_kwargs = {"tokenizer_name": args.tokenizer_path} if args.tokenizer_path else {}
        datasets.append(CREMADDataset(args.cremad_root, split="all", **tok_kwargs))
    if args.iemocap_root:
        train_sessions = tuple(int(s.strip()) for s in args.iemocap_train_sessions.split(","))
        datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=list(train_sessions)))
    if args.mustard_root:
        datasets.append(MUStARDDataset(args.mustard_root, split="all"))
    if args.meld_root:
        datasets.append(MELDDataset(args.meld_root, split="all"))

    if not datasets:
        raise ValueError("Provide at least one dataset root")

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    num_samples = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=supports_pin_memory(args.device),
        collate_fn=collate_fn,
        **({
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": True,
        } if args.num_workers > 0 else {}),
    )

    logger.info(f"Pre-computing embeddings for {num_samples} samples (device={args.device})")

    # Build model with frozen encoders only (no trainable components needed)
    from models.conflictnet import ConflictNet

    model = ConflictNet(
        audio_encoder_name=args.audio_encoder,
        embed_dim=args.embed_dim,
        use_speaker_norm=True,
        use_temporal=False,
        use_cross_attn_injection=False,
        use_speaker_adaptive_threshold=False,
        use_baseline_subtract=False,
        use_word_divergence=False,
        use_swap_pretraining=False,
        lora_r=16,
    ).to(args.device)
    model.eval()

    non_block = supports_non_blocking(args.device)

    # Storage
    all_audio = torch.zeros(num_samples, args.embed_dim)
    all_text = torch.zeros(num_samples, args.embed_dim)
    all_speaker = torch.zeros(num_samples, args.embed_dim)
    all_labels = torch.zeros(num_samples, 3)
    all_severity = torch.zeros(num_samples, 1)
    all_binary = torch.zeros(num_samples)
    all_has_real_type = torch.zeros(num_samples, dtype=torch.bool)
    all_ids: list[str] = []
    all_speaker_ids: list[str] = []

    start = 0
    for batch in loader:
        batch_gpu = {
            k: v.to(args.device, non_blocking=non_block) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        audio_embed, text_embed, speaker_feat, _, _ = model.encode(
            audio=batch_gpu["audio"],
            input_ids=batch_gpu["input_ids"],
            attention_mask=batch_gpu["attention_mask"],
            audio_attention_mask=batch_gpu.get("audio_attention_mask"),
            prosody_z=batch_gpu.get("prosody_z"),
        )
        B = audio_embed.size(0)
        all_audio[start:start+B] = audio_embed.cpu()
        all_text[start:start+B] = text_embed.cpu()
        all_speaker[start:start+B] = speaker_feat.cpu()
        all_labels[start:start+B] = batch["conflict_type_labels"]
        all_severity[start:start+B] = batch["severity"]
        all_binary[start:start+B] = batch["conflict_binary"]
        all_has_real_type[start:start+B] = batch.get("has_real_type_labels", torch.zeros(B, dtype=torch.bool))
        all_ids.extend(batch.get("utterance_id", [f"idx_{i}" for i in range(start, start+B)]))
        all_speaker_ids.extend(batch.get("speaker_ids", [f"spk_{i}" for i in range(start, start+B)]))
        start += B

        if (start // batch_size) % 50 == 0:
            logger.info(f"  [{start}/{num_samples}]")

        # Checkpoint every 500 batches (1000 samples)
        if start > 0 and (start // batch_size) % 500 == 0:
            _save_checkpoint(args.output_dir, all_audio, all_text, all_speaker,
                             all_labels, all_severity, all_binary, all_has_real_type, all_ids, all_speaker_ids,
                             num_samples, args.embed_dim, args.audio_encoder,
                             start, logger)

    _save_checkpoint(args.output_dir, all_audio, all_text, all_speaker,
                     all_labels, all_severity, all_binary, all_has_real_type, all_ids, all_speaker_ids,
                     num_samples, args.embed_dim, args.audio_encoder,
                     num_samples, logger)


def _save_checkpoint(out_dir, audio_embed, text_embed, speaker_feat,
                     labels, severity, binary, has_real_type, ids, speaker_ids, num_samples, embed_dim,
                     audio_encoder, processed, logger):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "audio_embed": audio_embed,
        "text_embed": text_embed,
        "speaker_feat": speaker_feat,
        "conflict_type_labels": labels,
        "severity": severity,
        "conflict_binary": binary,
        "has_real_type_labels": has_real_type,
        "utterance_ids": ids,
        "speaker_ids": speaker_ids,
        "num_samples": num_samples,
        "embed_dim": embed_dim,
        "audio_encoder": audio_encoder,
    }, out_dir / "embeddings.pt")
    logger.info(f"  [{processed}/{num_samples}] saved checkpoint to {out_dir / 'embeddings.pt'}")
    logger.info(f"    audio_embed:   {tuple(audio_embed.shape)}")


if __name__ == "__main__":
    main()
