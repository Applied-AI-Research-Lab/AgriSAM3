#!/bin/bash
# Evaluate Single Experiment
# Usage: bash experiments/evaluate_single.sh <experiment_name>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash experiments/evaluate_single.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    echo "  exp01_fruit_ripeness"
    echo "  exp02_crop_weed"
    echo "  exp03_disease"
    echo "  exp04_multi_crop"
    echo "  exp05_pests"
    echo "  exp06_greenhouse"
    exit 1
fi

EXPERIMENT=$1
CONFIG="configs/${EXPERIMENT}.yml"
CHECKPOINT="outputs/${EXPERIMENT}/checkpoints/best_model.pth"
OUTPUT="outputs/${EXPERIMENT}/evaluation"

echo "========================================"
echo "Single Experiment Evaluation"
echo "Experiment: $EXPERIMENT"
echo "========================================"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please train the experiment first."
    exit 1
fi

mkdir -p "$OUTPUT"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT"
echo ""

echo "Starting evaluation..."
echo "========================================"

python src/eval/experiment_evaluator.py \
    --mode single \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --visualize

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results:"
echo "  - Metrics: $OUTPUT/metrics.json"
echo "  - Visualizations: $OUTPUT/visualizations/"
echo "========================================"
