#!/usr/bin/env bash
# train_rtx_pro_parallel_phaseB.sh — Phase B: 2 parallel ablations on 1 RTX PRO 6000
# Run 3: No temporal context ablation
# Run 5: ConflictNet-mini baseline
# Expected: ~6 hrs, ~₹840 @ ₹140/hr

set -euo pipefail

# === CONFIGURATION ===
GPU_ID=0
CACHE_PATH="cached_embeddings/embeddings.pt"
PROSODY_STATS="prosody_stats.json"
EPOCHS=25
BATCH_SIZE=24
GRAD_ACCUM=2
LR=1.5e-5
LORA_R=16
NUM_WORKERS=2
SEED=42
MEMORY_FRACTION=0.45

AUDIO_ENCODER="wavlm_large"
EMBED_DIM=512

echo "======================================"
echo "  RTX PRO 6000 - Phase B (Parallel)"
echo "======================================"
echo "GPU: $GPU_ID (2 processes, $MEMORY_FRACTION each)"
echo "Run 3: No temporal context ablation"
echo "Run 5: ConflictNet-mini baseline"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run 3: No temporal context - background
python scripts/train.py \
    --cache_path "$CACHE_PATH" \
    --prosody_stats "$PROSODY_STATS" \
    --output_dir "checkpoints/run3_no_temporal" \
    --use_cached \
    --audio_encoder "$AUDIO_ENCODER" \
    --embed_dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --separation_lambda 0.1 \
    --lora_r "$LORA_R" \
    --bf16 \
    --amp \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --memory_fraction "$MEMORY_FRACTION" \
    --run_name "no_temporal" \
    --no_temporal \
    --no_cross_attn_injection &

PID3=$!

# Run 5: ConflictNet-mini baseline - background
python scripts/train.py \
    --cache_path "$CACHE_PATH" \
    --prosody_stats "$PROSODY_STATS" \
    --output_dir "checkpoints/run5_baseline" \
    --use_cached \
    --audio_encoder "$AUDIO_ENCODER" \
    --embed_dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --separation_lambda 0.1 \
    --lora_r "$LORA_R" \
    --bf16 \
    --amp \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --memory_fraction "$MEMORY_FRACTION" \
    --run_name "baseline_mini" \
    --no_speaker_adaptive_threshold \
    --no_baseline_subtract \
    --no_word_divergence &

PID5=$!

echo "Started Run 3 (PID: $PID3)"
echo "Started Run 5 (PID: $PID5)"
echo "Waiting for both to complete..."

wait $PID3
STATUS3=$?
wait $PID5
STATUS5=$?

if [ $STATUS3 -eq 0 ] && [ $STATUS5 -eq 0 ]; then
    echo "Phase B completed successfully!"
else
    echo "Phase B had failures (Run3: $STATUS3, Run5: $STATUS5)"
    exit 1
fi
