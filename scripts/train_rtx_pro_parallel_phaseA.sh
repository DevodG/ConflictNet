#!/usr/bin/env bash
# train_rtx_pro_parallel_phaseA.sh — Phase A: 2 parallel runs on 1 RTX PRO 6000
# Run 1: Full ConflictNet (canonical)
# Run 2: No separation wall ablation
# Expected: ~10 hrs, ~₹1,400 @ ₹140/hr

set -euo pipefail

# === CONFIGURATION ===
GPU_ID=0
CACHE_PATH="cached_embeddings/embeddings.pt"
PROSODY_STATS="prosody_stats.json"
EPOCHS=30
PRETRAIN_EPOCHS=3
BATCH_SIZE=24
GRAD_ACCUM=2
LR=1.5e-5
WARMUP=500
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
LORA_R=16
NUM_WORKERS=2
SEED=42
MEMORY_FRACTION=0.45

AUDIO_ENCODER="wavlm"
EMBED_DIM=256

echo "======================================"
echo "  RTX PRO 6000 - Phase A (Parallel)"
echo "======================================"
echo "GPU: $GPU_ID (2 processes, $MEMORY_FRACTION each)"
echo "Run 1: Full ConflictNet (canonical)"
echo "Run 2: No separation wall ablation"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run 1: Canonical - background
python scripts/train.py \
    --cache_path "$CACHE_PATH" \
    --prosody_stats "$PROSODY_STATS" \
    --output_dir "checkpoints/run1_full" \
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
    --separation_lambda 0.1 \
    --lora_r "$LORA_R" \
    --bf16 \
    --amp \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --memory_fraction "$MEMORY_FRACTION" \
    --run_name "canonical" \
    --target_f1 0.78 \
    --max_retries 2 \
    --resume_epochs 10 &

PID1=$!

# Run 2: No separation wall - background
python scripts/train.py \
    --output_dir "checkpoints/run2_no_sep" \
    --use_cached \
    --audio_encoder "$AUDIO_ENCODER" \
    --embed_dim "$EMBED_DIM" \
    --epochs 25 \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --separation_lambda 0.0 \
    --bf16 \
    --amp \
    --memory_fraction "$MEMORY_FRACTION" \
    --run_name "no_separation" &

PID2=$!

echo "Started Run 1 (PID: $PID1)"
echo "Started Run 2 (PID: $PID2)"
echo "Waiting for both to complete..."

wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo "Phase A completed successfully!"
else
    echo "Phase A had failures (Run1: $STATUS1, Run2: $STATUS2)"
    exit 1
fi
