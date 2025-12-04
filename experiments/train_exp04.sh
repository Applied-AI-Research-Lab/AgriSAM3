#!/bin/bash
# Training script for Experiment 04: Multi-Crop Field Segmentation
# Dataset: Agriculture-Vision (aerial imagery)
# Domain: Field boundary and crop type segmentation

set -e

echo "========================================"
echo "Experiment 04: Multi-Crop Field Segmentation"
echo "Dataset: Agriculture-Vision (aerial)"
echo "========================================"

EXPERIMENT="exp04_multi_crop"
CONFIG="configs/exp04_multi_crop.yml"
OUTPUT_DIR="outputs/${EXPERIMENT}"
PREV_CHECKPOINT="outputs/exp03_disease/checkpoints/best_model.pth"

if [ ! -f "$PREV_CHECKPOINT" ]; then
    echo "Warning: Previous experiment checkpoint not found: $PREV_CHECKPOINT"
    echo "Training from pretrained SAM3 instead."
    RESUME_FROM="pretrained"
else
    echo "Using progressive training from Experiment 03"
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
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
