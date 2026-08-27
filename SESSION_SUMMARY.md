# Session Summary — Pre-compute Pipeline for ConflictNet v2

## Goal
Train ConflictNet v2 locally on M2 Mac with pre-computed embeddings for speedup.

## State Before Session
- ConflictNet v2 fully implemented (fusion, classifier, contrastive, swap, focal loss, ECE metrics)
- M2-optimized auto-config (`mps_optimized_config()`) in `models/device_utils.py`
- All 131 tests passing
- `current_code/` mirror in sync
- Pre-compute approach designed but not yet implemented

## What Was Built

### 1. `scripts/precompute_embeddings.py`
One-time script to cache WavLM + DeBERTa + ECAPA encoder outputs to disk.
- Processes all 7442 CREMA-D samples through `ConflictNet.encode()`
- Saves `embeddings.pt` with stacked tensors (audio_embed, text_embed, speaker_feat, labels)
- Includes checkpointing every 500 batches

### 2. `scripts/precompute_incremental.py` (current approach)
Improved version with:
- Warmup pass (trigger MPS kernel compilation on 1 batch before main loop)
- Checkpoint resume (load existing `embeddings.pt` and skip already-processed)
- Per-batch error handling (skip bad batches)
- Configurable checkpoint interval (`--chunk_size`)
- Logging at every 10 batches

### 3. `data/cached_dataset.py`
`CachedEmbeddingDataset` — serves pre-computed embeddings for training:
- Loads stacked tensors from `embeddings.pt`
- Deterministic train/val split via `torch.randperm` with seed
- `cached_collate_fn` for simple stacking (no variable-length padding needed)

### 4. `models/cached_conflictnet.py`
Lightweight model for training on pre-computed embeddings:
- Reuses all trainable modules from ConflictNet (fusion gate, classifier, contrastive loss, swap objective, multi-task loss)
- Takes `(audio_embed, text_embed, speaker_feat)` directly — no WavLM/DeBERTa/ECAPA
- ~809K trainable params (vs 435M total in full ConflictNet)
- Verified via `--dry_run` with synthetic cached embeddings

### 5. `data/datasets.py` additions
- CREMADataset: added `split="all"` support for precompute (returns all 7442 samples)

### 6. `scripts/train.py` additions
- `--use_cached` flag: switches to CachedConflictNet + CachedEmbeddingDataset
- `--cache_path`: path to `embeddings.pt`
- Dry-run mode adapted for cached model (uses synthetic embeddings)
- When `--use_cached`, forces `num_workers=0, pin_memory=False`

## Precompute Run Results

### Performance
| Phase | Time/batch | Notes |
|-------|-----------|-------|
| Model init | ~8s | WavLM + DeBERTa + ECAPA loading |
| Model → MPS | ~2.5s | 1.2 GB allocated |
| Warmup batch | ~6s | Includes MPS kernel compilation |
| Steady state (first 30 min) | ~2s/batch | 19 batches/min → 3.2 hr total |
| After 1.5 hr | ~4 min/batch | Severe degradation — likely MPS thermal/memory issue |

### MPS Degradation
After ~1250 batches (2500 samples, ~33% of data), processing slowed from 2s/batch to 4+ min/batch. Causes:
1. **M2 MacBook Air is fanless** — thermal throttling under sustained GPU load
2. **Memory fragmentation** — MPS allocator doesn't release memory between batches
3. **GPU synchronization overhead** — grows with number of queued operations

### Current Progress
- **2500/7442 samples cached** (~34%)
- Checkpoint saved at: `cached_embeddings/embeddings.pt` (23 MB)
- Process killed at ~23:10 after degradation made it impractical
- Screen session still exists but process terminated

## To Continue Later

### Option A: Finish Precompute via Chunked Restarts (Recommended)
Run in 200-batch chunks with process restarts to reset MPS state:
```bash
# Each chunk processes 200 batches, saves, exits
# The script auto-skips already-saved samples via resume
rm -f cached_embeddings/embeddings.pt  # only if starting fresh
python scripts/precompute_incremental.py --chunk_size 200
```
If current checkpoint has partial data, the script will resume automatically.

**Expected**: 3721 batches ÷ 200 batches/chunk = ~19 chunks × ~7 min/chunk = ~2.2 hr total

### Option B: Train on Existing 34% Data
You can train `CachedConflictNet` on what's already cached:
```bash
python scripts/train.py --use_cached --cache_path cached_embeddings/embeddings.pt --epochs 15
```
But this only uses 2500/7442 samples (incomplete dataset).

### Option C: Cloud GPU
For full-scale training, use `.kaggle/train_conflictnet.py` or `lightning_train.py` with GPU.

## Files Changed/Created This Session
| File | Change |
|------|--------|
| `scripts/precompute_embeddings.py` | New — initial precompute script with checkpointing |
| `scripts/precompute_incremental.py` | New — improved incremental version with warmup + resume |
| `scripts/train.py` | Added `--use_cached`, `--cache_path` flags; adapted dry_run for cached model |
| `data/cached_dataset.py` | New — `CachedEmbeddingDataset` + `cached_collate_fn` |
| `data/datasets.py` | Added `split="all"` to CREMADataset |
| `models/cached_conflictnet.py` | New — lightweight model (809K params) for training on cached embeddings |
