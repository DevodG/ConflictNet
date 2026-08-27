#!/usr/bin/env bash
# train_rtx_pro_canonical.sh — Single canonical run on RTX PRO 6000
# Expected: ~10 hrs, ~₹1,400 @ ₹140/hr

set -euo pipefail

# === CONFIGURATION ===
GPU_ID=0
CACHE_PATH="cached_embeddings/embeddings.pt"
PROSODY_STATS="prosody_stats.json"
OUTPUT_DIR="checkpoints/run1_full"
EPOCHS=30
PRETRAIN_EPOCHS=3
BATCH_SIZE=24
GRAD_ACCUM=2
LR=1.5e-5
WARMUP=500
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
SEP_LAMBDA=0.1
LORA_R=16
NUM_WORKERS=2
SEED=42

# Audio encoder
AUDIO_ENCODER="wavlm"
EMBED_DIM=256

# Data roots (adjust for your remote machine)
IEMOCAP_ROOT="/data/iemocap"
MUSTARD_ROOT="/data/mustard"
CREMAD_ROOT="/data/cremad"
MELD_ROOT="/data/meld"

echo "======================================"
echo "  RTX PRO 6000 - Canonical Training"
echo "======================================"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS (pretrain: $PRETRAIN_EPOCHS)"
echo "Batch: $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "LR: $LR"
echo "Mixed precision: BF16"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

python scripts/train.py \
    --cache_path "$CACHE_PATH" \
    --prosody_stats "$PROSODY_STATS" \
    --output_dir "$OUTPUT_DIR" \
    --use_cached \
    --audio_encoder "$AUDIO_ENCODER" \
    --embed_dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --focal_loss_gamma "$FOCAL_GAMMA" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --separation_lambda "$SEP_LAMBDA" \
    --lora_r "$LORA_R" \
    --bf16 \
    --amp \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --memory_fraction 1.0 \
    --run_name "canonical" \
    --target_f1 0.78 \
    --max_retries 2 \
    --resume_epochs 10

echo "Training complete. Check $OUTPUT_DIR for best_model.safetensors"
