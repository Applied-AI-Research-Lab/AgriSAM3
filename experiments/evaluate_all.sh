#!/bin/bash
# Evaluate Universal Agricultural Model
# Tests performance across all 6 experimental domains

set -e

echo "========================================"
echo "AgriSAM3 Universal Model Evaluation"
echo "========================================"

# Universal model checkpoint
UNIVERSAL_CHECKPOINT="outputs/exp06_greenhouse/checkpoints/best_model.pth"

# Check if model exists
if [ ! -f "$UNIVERSAL_CHECKPOINT" ]; then
    echo "Error: Universal model not found at: $UNIVERSAL_CHECKPOINT"
    echo "Please run the training pipeline first: bash experiments/train_universal.sh"
    exit 1
fi

# Output directory
EVAL_OUTPUT="outputs/universal_evaluation"
mkdir -p "$EVAL_OUTPUT"

# Config files
CONFIGS=(
    "configs/exp01_fruit_ripeness.yml"
    "configs/exp02_crop_weed.yml"
    "configs/exp03_disease.yml"
    "configs/exp04_multi_crop.yml"
    "configs/exp05_pests.yml"
    "configs/exp06_greenhouse.yml"
)

echo ""
echo "Configuration:"
echo "  Model: $UNIVERSAL_CHECKPOINT"
echo "  Experiments: ${#CONFIGS[@]}"
echo "  Output: $EVAL_OUTPUT"
echo ""

# Run universal evaluation
echo "Starting universal evaluation..."
echo "========================================"

python src/eval/experiment_evaluator.py \
    --mode universal \
    --checkpoint "$UNIVERSAL_CHECKPOINT" \
    --output "$EVAL_OUTPUT" \
    --experiment_configs "${CONFIGS[@]}" \
    --visualize

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: $EVAL_OUTPUT"
echo ""
echo "Summary files:"
echo "  - universal_results.json: Aggregate metrics"
echo "  - exp01_fruit_ripeness/metrics.json: Fruit ripeness results"
echo "  - exp02_crop_weed/metrics.json: Crop-weed results"
echo "  - exp03_disease/metrics.json: Disease detection results"
echo "  - exp04_multi_crop/metrics.json: Multi-crop field results"
echo "  - exp05_pests/metrics.json: Pest detection results"
echo "  - exp06_greenhouse/metrics.json: Greenhouse results"
echo ""
echo "Visualizations:"
for config in "${CONFIGS[@]}"; do
    exp_name=$(basename "$config" .yml)
    echo "  - $EVAL_OUTPUT/$exp_name/visualizations/"
done
echo "========================================"
