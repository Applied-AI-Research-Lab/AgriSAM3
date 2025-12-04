#!/bin/bash
# Training script for Experiment 05: Insect Pest Segmentation
# Dataset: IP102 (102 agricultural pest species)
# Domain: Pest detection and species identification

set -e

echo "========================================"
echo "Experiment 05: Insect Pest Segmentation"
echo "Dataset: IP102 (102 pest species)"
echo "========================================"

EXPERIMENT="exp05_pests"
CONFIG="configs/exp05_pests.yml"
OUTPUT_DIR="outputs/${EXPERIMENT}"
PREV_CHECKPOINT="outputs/exp04_multi_crop/checkpoints/best_model.pth"

if [ ! -f "$PREV_CHECKPOINT" ]; then
    echo "Warning: Previous experiment checkpoint not found: $PREV_CHECKPOINT"
    echo "Training from pretrained SAM3 instead."
    RESUME_FROM="pretrained"
else
    echo "Using progressive training from Experiment 04"
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
