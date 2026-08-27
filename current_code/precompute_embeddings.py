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
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--audio_encoder", type=str, default="wavlm",
                   choices=["emotion2vec", "wavlm", "wav2vec2"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
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

    from data.augmentation import AudioAugmentor

    # No augmentation during pre-compute (augmentation applied per-epoch, not here)
    collate_fn = make_collate_fn()

    datasets = []
    if args.cremad_root:
        tok_kwargs = {"tokenizer_name": args.tokenizer_path} if args.tokenizer_path else {}
        datasets.append(CREMADDataset(args.cremad_root, split="all", **tok_kwargs))
    if args.iemocap_root:
        datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[1, 2, 3, 4, 5]))
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
        num_workers=0,
        collate_fn=collate_fn,
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

    from models.device_utils import supports_non_blocking
    non_block = supports_non_blocking(args.device)

    # Storage
    all_audio = torch.zeros(num_samples, args.embed_dim)
    all_text = torch.zeros(num_samples, args.embed_dim)
    all_speaker = torch.zeros(num_samples, args.embed_dim)
    all_labels = torch.zeros(num_samples, 3)
    all_severity = torch.zeros(num_samples, 1)
    all_binary = torch.zeros(num_samples)
    all_ids: list[str] = []

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
        all_ids.extend(batch.get("utterance_id", [f"idx_{i}" for i in range(start, start+B)]))
        start += B

        if (start // batch_size) % 50 == 0:
            logger.info(f"  [{start}/{num_samples}]")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "audio_embed": all_audio,
        "text_embed": all_text,
        "speaker_feat": all_speaker,
        "conflict_type_labels": all_labels,
        "severity": all_severity,
        "conflict_binary": all_binary,
        "utterance_ids": all_ids,
        "num_samples": num_samples,
        "embed_dim": args.embed_dim,
        "audio_encoder": args.audio_encoder,
    }, out_dir / "embeddings.pt")

    logger.info(f"Saved {num_samples} embeddings to {out_dir / 'embeddings.pt'}")
    logger.info(f"  audio_embed:   {tuple(all_audio.shape)}")
    logger.info(f"  text_embed:    {tuple(all_text.shape)}")
    logger.info(f"  speaker_feat:  {tuple(all_speaker.shape)}")
    logger.info(f"  labels:        {tuple(all_labels.shape)}")


if __name__ == "__main__":
    main()
