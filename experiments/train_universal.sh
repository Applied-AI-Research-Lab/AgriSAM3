#!/bin/bash
# Universal Training Pipeline
# Runs all 6 experiments in sequence with progressive training
# Creates universal agricultural segmentation model

set -e

echo "========================================"
echo "AgriSAM3 Universal Training Pipeline"
echo "Progressive Training Across 6 Experiments"
echo "========================================"

START_TIME=$(date +%s)

# Experiments in order
EXPERIMENTS=(
    "exp01_fruit_ripeness"
    "exp02_crop_weed"
    "exp03_disease"
    "exp04_multi_crop"
    "exp05_pests"
    "exp06_greenhouse"
)

echo ""
echo "Training Schedule:"
for i in "${!EXPERIMENTS[@]}"; do
    echo "  $((i+1)). ${EXPERIMENTS[$i]}"
done
echo ""

# Ask for confirmation
read -p "This will take approximately 35 hours. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=$((i+1))
    EXP_NAME="${EXPERIMENTS[$i]}"
    
    echo ""
    echo "========================================"
    echo "Stage $EXP_NUM/6: $EXP_NAME"
    echo "========================================"
    
    # Run training script
    bash "experiments/train_exp0${EXP_NUM}.sh"
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Stage $EXP_NUM completed successfully"
    else
        echo "✗ Stage $EXP_NUM failed"
        exit 1
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "Universal Training Pipeline Complete!"
echo "========================================"
echo "Total training time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Universal model location:"
echo "  outputs/exp06_greenhouse/checkpoints/best_model.pth"
echo ""
echo "This model can segment:"
echo "  ✓ Fruits (ripeness assessment)"
echo "  ✓ Crops and weeds"
echo "  ✓ Plant diseases"
echo "  ✓ Agricultural fields (aerial view)"
echo "  ✓ Insect pests"
echo "  ✓ Greenhouse environments"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: bash experiments/evaluate_all.sh"
echo "  2. Test on your own data: python src/inference/predict.py"
echo "========================================"
