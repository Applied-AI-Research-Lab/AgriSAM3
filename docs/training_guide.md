# AgriSAM3 Training Guide

Complete guide to fine-tuning SAM3 for agricultural segmentation tasks.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Running Experiments](#running-experiments)
6. [Progressive Training Strategy](#progressive-training-strategy)
7. [Monitoring Training](#monitoring-training)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

Train all experiments with progressive learning:

```bash
# Full pipeline (takes ~35 hours on A100 GPU)
bash experiments/train_universal.sh

# Or train individual experiments
bash experiments/train_exp01.sh  # Fruit ripeness
bash experiments/train_exp02.sh  # Crop-weed discrimination
# ... etc
```

Evaluate the universal model:

```bash
bash experiments/evaluate_all.sh
```

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 24GB VRAM (RTX 3090, RTX 4090)
- RAM: 32GB system memory
- Storage: 200GB free space for datasets and outputs

**Recommended:**
- GPU: NVIDIA A100 (40GB or 80GB)
- RAM: 64GB system memory
- Storage: 500GB SSD

### Software Requirements

```bash
# Python 3.9+
python --version

# CUDA 11.8+ (for PyTorch 2.0+)
nvidia-smi

# Install dependencies
pip install -r requirements.txt

# Install SAM3
pip install git+https://github.com/facebookresearch/sam3.git
```

### Download SAM3 Pretrained Checkpoint

```bash
# Create checkpoint directory
mkdir -p checkpoints

# Download SAM3 base model (update URL when available)
wget -O checkpoints/sam3_base.pth https://path/to/sam3_pretrained.pth

# Update config files to point to this checkpoint
# In each config/*.yml file, set:
# model:
#   checkpoint: "checkpoints/sam3_base.pth"
```

---

## Dataset Preparation

### 1. Experiment 01: MinneApple (Fruit Ripeness)

**Download:**
```bash
# MinneApple dataset
mkdir -p data/minneapple
cd data/minneapple

# Download from: https://github.com/nicolaihaeni/MinneApple
git clone https://github.com/nicolaihaeni/MinneApple.git
# Follow their instructions to organize data

# Expected structure:
# data/minneapple/
#   ├── train/
#   │   ├── images/
#   │   └── annotations.json
#   ├── val/
#   │   ├── images/
#   │   └── annotations.json
#   └── test/
#       ├── images/
#       └── annotations.json
```

### 2. Experiment 02: DeepWeeds + Agriculture-Vision

**DeepWeeds:**
```bash
mkdir -p data/crop_weed
cd data/crop_weed

# Download from: https://github.com/AlexOlsen/DeepWeeds
# Dataset: https://www.kaggle.com/datasets/imsparsh/deepweeds
```

**Agriculture-Vision:**
```bash
# Download from: https://github.com/SHI-Labs/Agriculture-Vision
# Registration required: https://www.agriculture-vision.com/
```

### 3. Experiment 03: PlantDoc + PlantVillage

**PlantDoc:**
```bash
mkdir -p data/plant_disease
cd data/plant_disease

# Download from: https://github.com/pratikkayal/PlantDoc-Dataset
```

**PlantVillage:**
```bash
# Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
# Or: https://github.com/spMohanty/PlantVillage-Dataset
```

### 4. Experiment 04: Agriculture-Vision (Aerial)

**Agriculture-Vision:**
```bash
mkdir -p data/agriculture_vision
cd data/agriculture_vision

# Download multi-crop segmentation data
# https://github.com/SHI-Labs/Agriculture-Vision
```

### 5. Experiment 05: IP102 (Insect Pests)

**IP102:**
```bash
mkdir -p data/ip102
cd data/ip102

# Download from: https://github.com/xpwu95/IP102
# Paper: https://arxiv.org/abs/1901.09129
```

### 6. Experiment 06: Greenhouse Dataset

**Custom dataset collection or simulation:**
```bash
mkdir -p data/greenhouse
cd data/greenhouse

# Option 1: Use existing greenhouse datasets
# - GreenHouse dataset: https://github.com/greenhouse-project
# - Indoor plant datasets from Roboflow Universe

# Option 2: Create synthetic data
# - Use simulation tools
# - Collect your own greenhouse images
```

### Data Format Conversion

If your data is not in COCO format:

```python
# Convert Pascal VOC to COCO
python scripts/convert_voc_to_coco.py \
    --voc_dir data/my_dataset/VOC \
    --output_json data/my_dataset/annotations.json

# Convert custom format
python scripts/convert_custom_to_coco.py \
    --input_dir data/my_dataset \
    --output_json data/my_dataset/annotations.json
```

---

## Training Configuration

### Understanding Config Files

Each experiment has a YAML config file in `configs/`:

```yaml
# configs/exp01_fruit_ripeness.yml

experiment:
  name: "exp01_fruit_ripeness"
  domain: "fruit_ripeness"
  concepts: ["ripeness", "fruit_detection", "color_based_classification"]

model:
  checkpoint: "checkpoints/sam3_base.pth"  # Path to pretrained SAM3

dataset:
  format: "coco"  # or "voc" or "json"
  images:
    train: "data/minneapple/train/images"
    val: "data/minneapple/val/images"
    test: "data/minneapple/test/images"
  annotations:
    train: "data/minneapple/train/annotations.json"
    val: "data/minneapple/val/annotations.json"
    test: "data/minneapple/test/annotations.json"

training:
  epochs: 50
  batch_size: 2  # Adjust based on GPU memory
  learning_rate: 0.00005
  gradient_accumulation: 4  # Effective batch size: 2*4=8
  
  # Module freezing
  freeze_vision_encoder: false  # Train vision encoder
  freeze_text_encoder: false    # Train text encoder
  freeze_tracker: true          # Always freeze (per SAM3 design)
  
  # Loss weights
  loss_weights:
    segmentation: 1.0
    grounding: 0.5
    ripeness: 0.3  # Domain-specific loss
```

### Key Hyperparameters

**Learning Rate:**
- Experiment 01 (from pretrained): `5e-5`
- Experiments 02-06 (progressive): `3e-5` to `1.5e-5` (decreasing)
- Rule of thumb: Reduce by ~30% when fine-tuning from previous experiment

**Batch Size:**
- SAM3 requires significant memory
- Single GPU (24GB): `batch_size=1` or `2`
- A100 (40GB): `batch_size=2` or `4`
- Use gradient accumulation to simulate larger batches

**Epochs:**
- Varies by dataset size
- Larger datasets: 35-40 epochs
- Smaller datasets: 50 epochs
- Monitor validation loss for early stopping

**Mixed Precision:**
- Always enabled (`mixed_precision: true`)
- Reduces memory by ~40%
- Speeds up training by ~2x

---

## Running Experiments

### Individual Experiment

```bash
# Train single experiment
bash experiments/train_exp01.sh

# Or directly with Python
python src/training/universal_finetune.py \
    --experiment exp01_fruit_ripeness \
    --config configs/exp01_fruit_ripeness.yml \
    --resume_from pretrained
```

### Progressive Training Pipeline

Run all experiments in sequence:

```bash
bash experiments/train_universal.sh
```

This script:
1. Trains Experiment 01 from pretrained SAM3
2. Trains Experiment 02 from Exp01 checkpoint
3. Continues through all 6 experiments
4. Creates universal agricultural model

**Estimated time:** 35 hours on A100 GPU

### Resume from Checkpoint

If training is interrupted:

```bash
# Resume from last checkpoint
python src/training/universal_finetune.py \
    --experiment exp03_disease \
    --config configs/exp03_disease.yml \
    --resume_from outputs/exp03_disease/checkpoints/latest.pth
```

### Custom Training

Create your own experiment:

1. **Create config file:**

```yaml
# configs/exp07_my_experiment.yml
experiment:
  name: "exp07_my_experiment"
  domain: "custom"
  concepts: ["my_concept_1", "my_concept_2"]

# ... rest of config
```

2. **Prepare dataset:**

Ensure COCO format or convert with `scripts/convert_to_coco.py`

3. **Run training:**

```bash
python src/training/universal_finetune.py \
    --experiment exp07_my_experiment \
    --config configs/exp07_my_experiment.yml \
    --resume_from pretrained
```

---

## Progressive Training Strategy

### Why Progressive Training?

Progressive training transfers knowledge across agricultural domains:

```
Pretrained SAM3
    ↓
Exp01: Fruit ripeness (learns fruit understanding)
    ↓
Exp02: Crop-weed (learns vegetation types)
    ↓
Exp03: Disease (learns symptom patterns)
    ↓
Exp04: Multi-crop (learns aerial perspective)
    ↓
Exp05: Pests (learns small object detection)
    ↓
Exp06: Greenhouse (learns indoor conditions)
    ↓
Universal Agricultural Model
```

### Benefits

1. **Knowledge Transfer:** Disease concepts help pest detection
2. **Data Efficiency:** Less data needed per domain
3. **Better Generalization:** Model learns diverse agricultural contexts
4. **Concept Composition:** Combines concepts (ripeness + health + species)

### Training Order

The order is carefully designed:

1. **Exp01 (Fruit):** Establishes fruit/object understanding
2. **Exp02 (Crop-weed):** Extends to vegetation classification
3. **Exp03 (Disease):** Adds health assessment concepts
4. **Exp04 (Aerial):** Changes perspective, tests generalization
5. **Exp05 (Pests):** Adds small object detection
6. **Exp06 (Greenhouse):** Indoor domain, completes coverage

---

## Monitoring Training

### TensorBoard

Monitor training in real-time:

```bash
# Start TensorBoard
tensorboard --logdir outputs/exp01_fruit_ripeness/logs

# View in browser
# http://localhost:6006
```

**Key metrics to watch:**
- **Train loss:** Should decrease steadily
- **Val loss:** Should decrease; if increasing, you're overfitting
- **Segmentation IoU:** Target > 0.70
- **Learning rate:** Follow cosine schedule

### Weights & Biases (Optional)

Enable in config:

```yaml
output:
  wandb: true
  wandb:
    project: "AgriSAM3"
    entity: "your-username"
    name: "exp01_fruit_ripeness"
```

Then train:

```bash
wandb login  # First time only
bash experiments/train_exp01.sh
```

### Training Logs

Check text logs:

```bash
# View latest training log
tail -f outputs/exp01_fruit_ripeness/logs/exp01_fruit_ripeness_*.log

# Check for errors
grep "ERROR" outputs/exp01_fruit_ripeness/logs/*.log
```

### Checkpoints

Checkpoints are saved every N epochs:

```
outputs/exp01_fruit_ripeness/checkpoints/
├── checkpoint_epoch_005.pth
├── checkpoint_epoch_010.pth
├── ...
├── best_model.pth        # Best validation loss
└── latest.pth            # Most recent
```

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
```yaml
# In config file, reduce:
training:
  batch_size: 1              # Reduce from 2
  gradient_accumulation: 8   # Increase from 4
  freeze_vision_encoder: true  # Freeze more modules
  freeze_text_encoder: true
```

Or use smaller resolution:
```yaml
training:
  resolution: 672  # Reduce from 1008
```

### Slow Training

**Issue:** Training too slow

**Solutions:**
- Enable mixed precision: `mixed_precision: true`
- Increase `num_workers`: `num_workers: 8`
- Use faster storage (SSD)
- Check GPU utilization: `nvidia-smi -l 1`

### Poor Performance

**Issue:** Low IoU or accuracy

**Check:**
1. **Data quality:** Visualize samples
2. **Learning rate:** May be too high/low
3. **Augmentation:** Ensure appropriate for domain
4. **Class imbalance:** Adjust loss weights

**Solutions:**
```yaml
training:
  learning_rate: 0.00003  # Reduce if unstable
  
  loss_weights:
    segmentation: 1.0
    grounding: 0.3        # Reduce if grounding dominates
```

### NaN Loss

**Issue:** Loss becomes NaN

**Causes:**
- Learning rate too high
- Gradient explosion
- Bad batch

**Solutions:**
```yaml
training:
  learning_rate: 0.00001  # Much lower
  gradient_clip: 0.5      # More aggressive clipping
  warmup_epochs: 5        # Longer warmup
```

### Dataset Loading Errors

**Issue:** Cannot load dataset

**Check:**
1. Data paths in config are correct
2. Annotation format matches config
3. Images and annotations aligned

```python
# Validate dataset
python scripts/validate_dataset.py \
    --config configs/exp01_fruit_ripeness.yml
```

---

## Best Practices

### 1. Start Small

Test with small subset before full training:

```python
# Modify config for quick test
training:
  epochs: 5
  
dataset:
  # Use only 100 samples
  subset_size: 100
```

### 2. Monitor Overfitting

```yaml
training:
  early_stopping:
    enabled: true
    patience: 10        # Stop if no improvement for 10 epochs
    min_delta: 0.0001
```

### 3. Save Resources

```yaml
training:
  save_frequency: 10     # Save less frequently
  keep_last_n: 2         # Keep only 2 recent checkpoints
```

### 4. Validate Data

Before training:
```bash
python scripts/visualize_dataset.py \
    --config configs/exp01_fruit_ripeness.yml \
    --num_samples 20
```

### 5. Version Control

```bash
# Save config alongside outputs
cp configs/exp01_fruit_ripeness.yml \
   outputs/exp01_fruit_ripeness/config_used.yml
```

---

## Advanced Topics

### Multi-GPU Training

```python
# In universal_finetune.py, enable DataParallel
# Will be added in future update

# For now, use single best GPU
export CUDA_VISIBLE_DEVICES=0
```

### Custom Loss Functions

Add domain-specific losses in `universal_finetune.py`:

```python
def compute_loss(self, predictions, batch):
    losses = {}
    
    # Standard losses
    losses['segmentation'] = self._segmentation_loss(...)
    losses['grounding'] = self._grounding_loss(...)
    
    # Your custom loss
    if 'my_custom_attribute' in predictions:
        losses['custom'] = self._my_custom_loss(...)
    
    losses['total'] = sum(losses.values())
    return losses
```

### Hyperparameter Search

```bash
# Create multiple configs with different hyperparameters
for lr in 0.00005 0.00003 0.00001; do
    # Modify config
    sed "s/learning_rate: .*/learning_rate: $lr/" \
        configs/exp01_fruit_ripeness.yml > \
        configs/exp01_lr${lr}.yml
    
    # Train
    python src/training/universal_finetune.py \
        --experiment exp01_lr${lr} \
        --config configs/exp01_lr${lr}.yml
done
```

---

## Next Steps

After training:

1. **Evaluate:** `bash experiments/evaluate_all.sh`
2. **Visualize:** Check `outputs/*/visualizations/`
3. **Deploy:** See `docs/deployment.md`
4. **Fine-tune:** Adjust for your specific use case

For inference on new images, see `docs/inference_guide.md`.
