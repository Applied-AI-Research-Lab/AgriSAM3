#!/bin/bash
# Training script for Experiment 06: Greenhouse Environment Segmentation
# Dataset: Custom greenhouse dataset
# Domain: Indoor agriculture and controlled environment

set -e

echo "========================================"
echo "Experiment 06: Greenhouse Segmentation"
echo "Dataset: Custom greenhouse dataset"
echo "========================================"

EXPERIMENT="exp06_greenhouse"
CONFIG="configs/exp06_greenhouse.yml"
OUTPUT_DIR="outputs/${EXPERIMENT}"
PREV_CHECKPOINT="outputs/exp05_pests/checkpoints/best_model.pth"

if [ ! -f "$PREV_CHECKPOINT" ]; then
    echo "Warning: Previous experiment checkpoint not found: $PREV_CHECKPOINT"
    echo "Training from pretrained SAM3 instead."
    RESUME_FROM="pretrained"
else
    echo "Using progressive training from Experiment 05"
    RESUME_FROM="$PREV_CHECKPOINT"
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Config: $CONFIG"
echo "  Resume from: $RESUME_FROM"
echo "  Output: $OUTPUT_DIR"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "Starting training..."
echo "========================================"

python src/training/universal_finetune.py \
    --experiment "$EXPERIMENT" \
    --config "$CONFIG" \
    --resume_from "$RESUME_FROM"

echo ""
echo "========================================"
echo "Training completed!"
echo "This is the final experiment in the progressive training pipeline."
echo "Universal agricultural model saved to: $OUTPUT_DIR"
echo "========================================"
