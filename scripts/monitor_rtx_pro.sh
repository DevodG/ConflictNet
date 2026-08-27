#!/usr/bin/env bash
# monitor_rtx_pro.sh — Monitor dual training runs on 1 RTX PRO 6000
# Usage: ./monitor_rtx_pro.sh [interval_seconds]

set -euo pipefail

INTERVAL=${1:-30}

echo "======================================"
echo "  RTX PRO 6000 Training Monitor"
echo "======================================"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== $(date) ==="
    echo ""
    
    # GPU utilization
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
        echo ""
    fi
    
    # Python processes
    echo "=== Python Training Processes ==="
    ps aux | grep -E "python.*train\.py" | grep -v grep || echo "No training processes found"
    echo ""
    
    # Disk usage
    echo "=== Checkpoint Sizes ==="
    du -sh checkpoints/*/ 2>/dev/null | sort -h || echo "No checkpoints yet"
    echo ""
    
    # Wandb status (if running)
    if command -v wandb &> /dev/null; then
        echo "=== WandB Status ==="
        wandb status 2>/dev/null || echo "WandB not active"
        echo ""
    fi
    
    sleep $INTERVAL
done