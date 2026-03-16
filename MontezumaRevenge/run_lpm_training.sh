#!/bin/bash
# LPM Training Script for Montezuma's Revenge
# This script runs LPM training in the background with proper logging

# Create log directory if it doesn't exist
mkdir -p ./logs/MR/lpm

# Set log file with timestamp
LOG_FILE="./logs/MR/lpm/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="./logs/MR/lpm/train.pid"

echo "Starting LPM training..."
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"

# Run training in background
nohup python -u main.py \
    --env-name MontezumaRevengeNoFrameskip-v4 \
    --algo ppo-lpm \
    --use-gae \
    --lr 1e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 128 \
    --num-steps 128 \
    --num-mini-batch 8 \
    --ppo-epoch 3 \
    --log-interval 1 \
    --entropy-coef 0.001 \
    --num-env-steps 200000000 \
    --log-dir ./logs/MR/lpm \
    --seed 1 \
    > "$LOG_FILE" 2>&1 &

# Save process ID
PID=$!
echo $PID > "$PID_FILE"

echo "Training started with PID: $PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if running: ps -p $PID"
echo "To stop training: kill $PID"

