#!/bin/bash
# Training script for Experiment 01: Fruit Ripeness Segmentation
# Dataset: MinneApple
# Domain: Fruit ripeness assessment

set -e  # Exit on error

echo "========================================"
echo "Experiment 01: Fruit Ripeness"
echo "Dataset: MinneApple"
echo "========================================"

# Configuration
EXPERIMENT="exp01_fruit_ripeness"
CONFIG="configs/exp01_fruit_ripeness.yml"
OUTPUT_DIR="outputs/${EXPERIMENT}"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration summary
echo ""
echo "Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "Warning: No GPU detected. Training will be slow."
    echo ""
fi

# Run training
echo "Starting training..."
echo "========================================"

python src/training/universal_finetune.py \
    --experiment "$EXPERIMENT" \
    --config "$CONFIG" \
    --resume_from pretrained

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
