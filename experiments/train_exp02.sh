#!/bin/bash
# Training script for Experiment 02: Crop-Weed Segmentation
# Datasets: DeepWeeds + Agriculture-Vision
# Domain: Crop-weed discrimination

set -e

echo "========================================"
echo "Experiment 02: Crop-Weed Segmentation"
echo "Datasets: DeepWeeds + Agriculture-Vision"
echo "========================================"

EXPERIMENT="exp02_crop_weed"
CONFIG="configs/exp02_crop_weed.yml"
OUTPUT_DIR="outputs/${EXPERIMENT}"
PREV_CHECKPOINT="outputs/exp01_fruit_ripeness/checkpoints/best_model.pth"

# Check if previous experiment checkpoint exists
if [ ! -f "$PREV_CHECKPOINT" ]; then
    echo "Warning: Previous experiment checkpoint not found: $PREV_CHECKPOINT"
    echo "Training from pretrained SAM3 instead of progressive training."
    RESUME_FROM="pretrained"
else
    echo "Using progressive training from Experiment 01"
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
