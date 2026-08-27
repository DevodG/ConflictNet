"""Incremental pre-compute with checkpointing and warmup.

Processes CREMA-D in small batches, saving after every 100 batches.
Resumes from existing checkpoint if available.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cremad_root", default="data/cremad")
    p.add_argument("--output_dir", default="cached_embeddings")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=100, help="Save checkpoint every N batches")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    from models.device_utils import resolve_device

    device = args.device or resolve_device()
    logger.info(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "embeddings.pt"

    # Dataset
    from data.datasets import CREMADDataset
    from data.datasets import make_collate_fn

    ds = CREMADDataset(args.cremad_root, split="all")
    logger.info(f"Dataset: {len(ds)} samples")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=make_collate_fn(),
    )

    # Model
    from models.conflictnet import ConflictNet

    logger.info("Building model...")
    t0 = time.time()
    model = ConflictNet(
        audio_encoder_name="wavlm",
        embed_dim=256,
        use_speaker_norm=True,
        use_temporal=False,
        use_cross_attn_injection=False,
        use_speaker_adaptive_threshold=False,
        use_baseline_subtract=False,
        use_word_divergence=False,
        use_swap_pretraining=False,
        lora_r=16,
    )
    logger.info(f"Model created ({time.time()-t0:.1f}s)")

    t0 = time.time()
    model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} ({time.time()-t0:.1f}s)")

    # Warmup: 1 batch to trigger MPS kernel compilation
    logger.info("Warmup: running 1 batch...")
    warmup_batch = next(iter(loader))
    warmup_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in warmup_batch.items()}
    with torch.no_grad():
        _, _, _, _, _ = model.encode(
            audio=warmup_gpu["audio"],
            input_ids=warmup_gpu["input_ids"],
            attention_mask=warmup_gpu["attention_mask"],
        )
    logger.info("Warmup complete")

    # Check for existing checkpoint
    start = 0
    all_audio, all_text, all_speaker, all_labels, all_severity, all_binary, all_ids = None, None, None, None, None, None, None
    if ckpt_path.exists():
        logger.info(f"Existing checkpoint found: {ckpt_path}, resuming...")
        data = torch.load(ckpt_path, weights_only=True)
        start = len(data["utterance_ids"])
        all_audio = data["audio_embed"]
        all_text = data["text_embed"]
        all_speaker = data["speaker_feat"]
        all_labels = data["conflict_type_labels"]
        all_severity = data["severity"]
        all_binary = data["conflict_binary"]
        all_ids = list(data["utterance_ids"])
        logger.info(f"Resumed with {start} samples already processed")

    # Skip past already-processed batches
    batch_idx = 0
    for batch_idx, batch in enumerate(loader):
        processed_samples = batch_idx * args.batch_size
        if processed_samples < start:
            continue
        break
    else:
        logger.info("All batches already processed!")
        return

    # Allocate storage on first batch
    N = len(ds)
    if all_audio is None:
        all_audio = torch.zeros(N, 256)
        all_text = torch.zeros(N, 256)
        all_speaker = torch.zeros(N, 256)
        all_labels = torch.zeros(N, 3)
        all_severity = torch.zeros(N, 1)
        all_binary = torch.zeros(N)
        all_ids = []

    # Process remaining batches
    batches_since_ckpt = 0
    for i, batch in enumerate(loader):
        idx = i * args.batch_size
        if idx < start:
            continue

        try:
            gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ae, te, sf, _, _ = model.encode(
                audio=gpu["audio"],
                input_ids=gpu["input_ids"],
                attention_mask=gpu["attention_mask"],
                audio_attention_mask=gpu.get("audio_attention_mask"),
                prosody_z=gpu.get("prosody_z"),
            )
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            continue

        B = ae.size(0)
        all_audio[idx:idx+B] = ae.cpu()
        all_text[idx:idx+B] = te.cpu()
        all_speaker[idx:idx+B] = sf.cpu()
        all_labels[idx:idx+B] = batch["conflict_type_labels"]
        all_severity[idx:idx+B] = batch["severity"]
        all_binary[idx:idx+B] = batch["conflict_binary"]
        all_ids.extend(batch.get("utterance_id", [f"idx_{j}" for j in range(idx, idx+B)]))

        batches_since_ckpt += 1
        if (i + 1) % 10 == 0:
            logger.info(f"  [{idx+B}/{N}] batch {i+1}")

        if batches_since_ckpt >= args.chunk_size:
            _save(ckpt_path, all_audio, all_text, all_speaker, all_labels,
                  all_severity, all_binary, all_ids, N)
            logger.info(f"  checkpoint saved ({idx+B}/{N})")
            batches_since_ckpt = 0

    _save(ckpt_path, all_audio, all_text, all_speaker, all_labels,
          all_severity, all_binary, all_ids, N)
    logger.info(f"Done! All {N} embeddings saved to {ckpt_path}")


def _save(path, audio, text, speaker, labels, severity, binary, ids, N):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "audio_embed": audio,
        "text_embed": text,
        "speaker_feat": speaker,
        "conflict_type_labels": labels,
        "severity": severity,
        "conflict_binary": binary,
        "utterance_ids": ids,
        "num_samples": N,
        "embed_dim": 256,
        "audio_encoder": "wavlm",
    }, path)


if __name__ == "__main__":
    main()
