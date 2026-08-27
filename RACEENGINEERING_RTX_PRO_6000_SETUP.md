# ConflictNet v2 — RTX PRO 6000 Training Setup for raceengineering.ai

## GPU Specification

**Required GPU: 1× NVIDIA RTX PRO 6000 (Blackwell, 96GB GDDR7)**

| Specification | Value |
|---------------|-------|
| Architecture | Blackwell (GB202) |
| VRAM | 96 GB GDDR7 |
| Memory Bandwidth | ~1.8 TB/s |
| FP16/BF16 TFLOPS | ~840 |
| FP8 TFLOPS | ~1,680 |
| CUDA Cores | ~18,432 |
| Tensor Cores | 576 (5th gen) |
| TDP | 600W |

**Why RTX PRO 6000:** 96GB VRAM fits 2 parallel training runs (43GB each) with buffer. Cost: ₹140/hr.

---

## Software Environment

```bash
# Base image: Ubuntu 22.04 LTS
# CUDA: 12.8 (required for Blackwell)
# cuDNN: 9.x
# NVIDIA Driver: 570+ (Blackwell support)

# Verify installation
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**PyTorch Installation:**
```bash
# For CUDA 12.8 (Blackwell)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## Dataset Acquisition (Pre-download before training)

| Dataset | Size | Download | Priority |
|---------|------|----------|----------|
| **CREMA-D** | 1.5 GB | Already at `data/cremad/AudioWAV/` | ✅ Ready |
| **MUStARD++** | 1.2 GB | `https://github.com/soujanyaporia/MUStARD` → `utterances_final.zip` | 🔴 Critical |
| **IEMOCAP** | 12 GB | USC license (see AUDIT_FINDINGS.md) | 🔴 Critical |
| **MELD** | 4 GB | `https://github.com/declare-lab/MELD` | 🟡 Recommended |
| **CMU-MOSEI** | 50 GB (audio+text) | `https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK` | 🟡 Skip video |

**Directory Structure on Remote:**
```
/data/
├── iemocap/          # IEMOCAP sessions 1-5
├── mustard/          # MUStARD++ utterances_final/
├── cremad/           # CREMA-D AudioWAV/ (already here)
├── meld/             # MELD train/dev/test
└── mosei/            # CMU-MOSEI (optional)
```

---

## Pre-computation Pipeline (Run Once)

### Step 1: Pre-compute Encoder Embeddings (~3 hrs, ₹420)
```bash
python scripts/precompute_embeddings.py \
    --iemocap_root /data/iemocap \
    --mustard_root /data/mustard \
    --cremad_root /data/cremad \
    --meld_root /data/meld \
    --audio_encoder wavlm_large \
    --output_dir cached_embeddings/
```
**Output:** `cached_embeddings/embeddings.pt` with keys:
- `audio_embed`: (N, 512) pre-computed WavLM-large embeddings
- `text_embed`: (N, 512) pre-computed DeBERTa-v3-base embeddings
- `speaker_feat`: (N, 512) speaker embeddings
- `conflict_type_labels`: (N, 3) multi-hot
- `severity`: (N, 1) float, NaN for proxy datasets
- `conflict_binary`: (N,) int
- `has_real_type_labels`: (N,) bool — True only for MUStARD++ & CASE
- `speaker_ids`: (N,) list of speaker IDs
- `utterance_ids`: (N,) list of utterance IDs

### Step 2: Pre-compute Prosody Stats (~15 min, negligible)
```bash
python scripts/compute_prosody_stats.py \
    --iemocap_root /data/iemocap \
    --mustard_root /data/mustard \
    --cremad_root /data/cremad \
    --meld_root /data/meld \
    --output_file prosody_stats.json
```
**Output:** 
- `prosody_stats.train.zscores.json` — train speaker z-scores
- `prosody_stats.val.zscores.json` — val speaker z-scores

---

## Training Phases

### Phase A: Canonical + No-Sep Ablation (Parallel, ~10 hrs, ₹1,400)
```bash
# Terminal 1: Run this script
chmod +x scripts/train_rtx_pro_parallel_phaseA.sh
./scripts/train_rtx_pro_parallel_phaseA.sh
```
**Runs:**
- `checkpoints/run1_full/` — Full ConflictNet (target F1: 0.78)
- `checkpoints/run2_no_sep/` — Separation wall λ=0

### Phase B: Temporal + Baseline Ablations (Parallel, ~6 hrs, ₹840)
```bash
# Terminal 1: Run this script
chmod +x scripts/train_rtx_pro_parallel_phaseB.sh
./scripts/train_rtx_pro_parallel_phaseB.sh
```
**Runs:**
- `checkpoints/run3_no_temporal/` — No temporal context
- `checkpoints/run5_baseline/` — ConflictNet-mini

### Phase C: Ensemble & Evaluation (~2 hrs, ₹280)
```bash
# Ensemble top 3
python scripts/ensemble.py \
    --checkpoints checkpoints/run1_full,checkpoints/run2_no_sep,checkpoints/run5_baseline \
    --cache_path cached_embeddings/embeddings.pt

# Generate LaTeX ablation table
python scripts/ablation_table.py \
    --runs run1_full,run2_no_sep,run3_no_temporal,run5_baseline \
    --output ablation_table.tex
```

---

## Monitoring (Open in separate terminals)

```bash
# Terminal 2: GPU monitor (30s interval)
chmod +x scripts/monitor_rtx_pro.sh
./scripts/monitor_rtx_pro.sh 30

# Terminal 3: Budget tracker
chmod +x scripts/budget_tracker.sh
./scripts/budget_tracker.sh
```

---

## Expected Wall Time & Cost

| Phase | Wall Time | Cost (₹) | What You Get |
|-------|-----------|----------|--------------|
| Pre-compute | ~3.25 hrs | 455 | Embeddings + prosody stats |
| Phase A | ~10 hrs | 1,400 | Canonical + no-sep ablation |
| Phase B | ~6 hrs | 840 | No-temporal + baseline-mini |
| Phase C | ~2 hrs | 280 | Ensemble + LaTeX table |
| **TOTAL** | **~21.25 hrs** | **₹2,975** | **Full ablation suite** |

---

## Key Configuration (from `configs/default.yaml`)

```yaml
audio_encoder: wavlm_large          # 315M params, best emotion F1
text_encoder: deberta-v3-base       # 140M params, 4x faster than large
embed_dim: 512                       # Wider for H200/RTX PRO 6000 capacity

training:
  batch_size: 24                     # 24 × 2 grad_accum = 48 effective
  gradient_accumulation_steps: 2
  lr: 1.5e-5
  epochs: 30
  pretrain_epochs: 3
  warmup_steps: 500
  focal_loss_gamma: 2.0
  label_smoothing: 0.1
  separation_lambda: 0.1            # Gap 9: separation wall
  lora: { enabled: true, r: 16, alpha: 32 }
  amp: true
  bf16: true                         # Blackwell native BF16
  num_workers: 2                     # Per process (4 total)

# Memory management for 2 parallel runs
memory_fraction: 0.45               # 43GB cap per process on 96GB
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **OOM on startup** | Reduce `batch_size` to 16, increase `gradient_accumulation_steps` to 3 |
| **BF16 not supported** | Remove `--bf16` flag, use `--amp` only (fp16) |
| **Slow data loading** | Increase `num_workers` to 4, ensure NVMe storage |
| **WandB not logging** | `export WANDB_PROJECT=conflictnet-v2` before training |
| **MUStARD++ audio missing** | Download `utterances_final.zip` from GitHub repo |
| **IEMOCAP license** | Ensure USC license file present, contact authors if needed |

---

## Quick Commands Reference

```bash
# Check GPU
nvidia-smi -l 5

# Check training logs
tail -f checkpoints/run1_full/train.log

# Resume from checkpoint
python current_code/train.py \
    --resume_from checkpoints/run1_full/best_model.safetensors \
    --output_dir checkpoints/run1_full \
    ... (other args same)

# Dry run (validate pipeline, no data needed)
python current_code/train.py --dry_run --use_cached --bf16 --amp

# Evaluate single checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/run1_full/best_model.safetensors \
    --cache_path cached_embeddings/embeddings.pt
```

---

## Paper-Ready Outputs

After Phase C, you'll have:
1. **`ablation_table.tex`** — LaTeX table for paper
2. **`checkpoints/ensemble/ensemble_predictions.pt`** — Ensemble predictions
3. **4 trained models** with full metrics in `best_model_meta.json`
4. **Cost receipt** from budget_tracker.sh

**Target F1 (surpassing literature):**
| Dataset | Literature | Our Target |
|---------|------------|------------|
| MUStARD++ sarcasm | 0.71-0.76 | **0.78-0.82** |
| IEMOCAP emotion | 0.65 | **0.70-0.74** |
| CMU-MOSEI sentiment | 0.83 | **0.85-0.87** |

---

## Support

- Check `AUDIT_FINDINGS.md` for known issues and fixes
- Review `ARCHITECTURE.md` for model details
- All fixes from audit applied: P0.1-P0.7, P1.1-P1.5, Gap 9 separation wall