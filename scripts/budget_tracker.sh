#!/usr/bin/env bash
# budget_tracker.sh — Real-time cost tracking for RTX PRO 6000 training

set -euo pipefail

RATE_PER_HOUR=140  # INR
START_TIME=$(date +%s)

echo "======================================"
echo "  Budget Tracker - RTX PRO 6000"
echo "======================================"
echo "Rate: ₹${RATE_PER_HOUR}/hr"
echo "Start: $(date)"
echo "Press Ctrl+C to stop and show final cost"
echo ""

while true; do
    sleep 60
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    HOURS=$(echo "scale=4; $ELAPSED / 3600" | bc -l)
    COST=$(echo "scale=2; $HOURS * $RATE_PER_HOUR" | bc -l)
    
    printf "\rElapsed: %02d:%02d:%02d  |  Cost: ₹%.2f  " \
        $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) \
        "$COST"
done

# On Ctrl+C
echo ""
echo ""
echo "======================================"
echo "  Final Cost Summary"
echo "======================================"
echo "Total time: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
printf "Total cost: ₹%.2f\n" "$COST"
echo "======================================"