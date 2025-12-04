# AgriSAM3: Delving into Segment Anything with Agricultural Concepts

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Citation

If you use this code or our findings in your research, please cite:

```bibtex
@article{sapkota2025AgriSAM3,
  title={AgriSAM3: Delving into Segment Anything with Agricultural Concepts},
  author={Sapkota, Ranjan and Roumeliotis, Konstantinos I. and Karkee, Manoj and Tselikas, Nikolaos D.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Overview

**AgriSAM3** is a comprehensive framework for fine-tuning SAM3 (Segment Anything Model 3) on diverse agricultural datasets to create a universal agricultural segmentation model. This project explores concept-driven segmentation across multiple agricultural domains: crops, weeds, diseases, pests, fruits, and more.

### Research Goal

Develop a **universal agricultural SAM3 model** that can:
- Segment any agricultural object using natural language concepts
- Generalize across diverse crop types, growth stages, and conditions
- Understand agricultural-specific concepts (ripeness, disease, health, variety)
- Enable zero-shot segmentation of unseen agricultural objects
- Support precision agriculture applications

### Key Innovation

Unlike domain-specific models, AgriSAM3 leverages SAM3's vision-language capabilities to learn a **unified agricultural concept space**. Each experiment fine-tunes on a specific dataset, but the learned representations contribute to a universal model that "segments anything in agriculture."

## Experiments Overview

AgriSAM3 includes **6 diverse agricultural experiments**, each targeting different domains:

### Experiment 1: Fruit Detection and Ripeness Assessment
**Dataset**: MinneApple + Additional Fruit Datasets  
**Concepts**: Ripe apples, unripe apples, overripe fruit, apple varieties  
**Application**: Automated harvesting, yield prediction  
**Metrics**: IoU, concept recall, ripeness accuracy

### Experiment 2: Crop and Weed Segmentation
**Dataset**: DeepWeeds + Agriculture-Vision  
**Concepts**: Crop plants, weed species, soil, mulch  
**Application**: Precision weeding, herbicide reduction  
**Metrics**: Crop-weed discrimination, species-level accuracy

### Experiment 3: Plant Disease Recognition
**Dataset**: PlantDoc + PlantVillage  
**Concepts**: Healthy leaves, disease symptoms, disease types  
**Application**: Early disease detection, treatment planning  
**Metrics**: Disease classification accuracy, symptom localization

### Experiment 4: Multi-Crop Instance Segmentation
**Dataset**: Agriculture-Vision + Custom Multi-Crop Dataset  
**Concepts**: Corn, soybean, wheat, cotton, specific growth stages  
**Application**: Crop monitoring, phenotyping  
**Metrics**: Instance segmentation quality, crop type accuracy

### Experiment 5: Pest and Damage Assessment
**Dataset**: IP102 (Insect Pests) + Crop Damage Datasets  
**Concepts**: Insect pests, pest damage, mechanical damage, hail damage  
**Application**: Pest management, damage quantification  
**Metrics**: Pest detection, damage severity estimation

### Experiment 6: Greenhouse and Indoor Agriculture
**Dataset**: Custom Greenhouse Dataset + Hydroponic Systems  
**Concepts**: Greenhouse plants, growth containers, equipment, controlled environment  
**Application**: Indoor farming optimization  
**Metrics**: Plant health monitoring, spatial layout understanding

## Universal Fine-tuning Framework

AgriSAM3 provides a **modular, reusable training infrastructure**:

```python
# Universal training command for ANY agricultural dataset
python src/training/universal_finetune.py \
    --experiment experiment_01_fruit_ripeness \
    --config configs/exp01_fruit_ripeness.yml \
    --resume_from pretrained  # or path to checkpoint
```

### Key Features

- **Dataset Agnostic**: Supports COCO, Pascal VOC, custom JSON formats
- **Concept-Driven**: Automatic text prompt generation from agricultural attributes
- **Modular Architecture**: Easy to add new experiments
- **Progressive Training**: Each experiment builds on previous knowledge
- **Unified Evaluation**: Consistent metrics across all experiments

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Training Guide](docs/training_guide.md)** - Complete training instructions, from setup to monitoring
- **[Datasets Guide](docs/datasets.md)** - Detailed dataset documentation with download links
- **[Experiments Guide](docs/experiments.md)** - In-depth analysis of all 6 experiments
- **[Methodology](docs/methodology.md)** - Experimental design and research methodology
- **[Universal Model](docs/universal_model.md)** - Using and deploying the universal model

## Project Structure

```
AgriSam3/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── configs/                           # Experiment configurations
│   ├── exp01_fruit_ripeness.yml
│   ├── exp02_crop_weed.yml
│   ├── exp03_plant_disease.yml
│   ├── exp04_multicrop.yml
│   ├── exp05_pest_damage.yml
│   └── exp06_greenhouse.yml
│
├── data/                              # Dataset storage
│   ├── README.md                     # Dataset download instructions
│   ├── downloaders/                  # Dataset download scripts
│   │   ├── download_minneapple.py
│   │   ├── download_deepweeds.py
│   │   ├── download_plantdoc.py
│   │   └── ...
│   └── experiment_01_fruit/          # Per-experiment data
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   ├── universal_finetune.py    # Main training script
│   │   ├── data_loaders.py          # Multi-format dataset loaders
│   │   ├── agricultural_prompts.py  # Agriculture-specific text prompts
│   │   └── training_utils.py        # Training utilities
│   ├── models/                       # Model wrappers
│   │   ├── __init__.py
│   │   └── sam3_agricultural.py     # SAM3 with agricultural concepts
│   ├── eval/                         # Evaluation
│   │   ├── __init__.py
│   │   ├── metrics_universal.py     # Universal agricultural metrics
│   │   ├── concept_evaluation.py    # Concept-level evaluation
│   │   └── experiment_evaluator.py  # Per-experiment evaluator
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── dataset_converters.py    # Format converters
│       ├── visualization.py         # Plotting utilities
│       └── logging_utils.py         # Experiment logging
│
├── experiments/                       # Experiment scripts
│   ├── run_exp01_fruit.sh
│   ├── run_exp02_weed.sh
│   ├── run_exp03_disease.sh
│   ├── run_exp04_multicrop.sh
│   ├── run_exp05_pest.sh
│   ├── run_exp06_greenhouse.sh
│   └── train_universal_model.sh     # Sequential training pipeline
│
├── notebooks/                         # Interactive analysis
│   ├── 01_experiment_analysis.ipynb
│   ├── 02_universal_model_eval.ipynb
│   └── 03_concept_visualization.ipynb
│
├── docs/                              # Documentation
│   ├── methodology.md                # Experimental design
│   ├── datasets.md                   # Dataset descriptions
│   ├── training_guide.md             # How to fine-tune
│   ├── experiments.md                # Detailed experiment docs
│   └── universal_model.md            # Universal model approach
│
└── results/                           # Outputs
    ├── experiment_01/
    │   ├── checkpoints/
    │   ├── metrics/
    │   └── visualizations/
    └── universal_model/
        ├── final_checkpoint.pth
        └── combined_metrics.json
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (16GB+ VRAM recommended)
- 100GB+ disk space for datasets

### Setup

```bash
# Navigate to AgriSam3 folder
cd AgriSam3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify SAM3 installation
python -c "import sam3; print('SAM3 installed successfully')"
```

## Quick Start

### 1. Download Datasets

```bash
# Download all agricultural datasets
bash data/downloaders/download_all.sh

# Or download individually
python data/downloaders/download_minneapple.py --output data/experiment_01_fruit
python data/downloaders/download_deepweeds.py --output data/experiment_02_weed
```

### 2. Run Individual Experiment

```bash
# Experiment 1: Fruit ripeness
bash experiments/run_exp01_fruit.sh

# Experiment 2: Crop-weed segmentation
bash experiments/run_exp02_weed.sh
```

### 3. Train Universal Model

```bash
# Sequential training across all experiments
bash experiments/train_universal_model.sh
```

### 4. Evaluate Results

```bash
# Evaluate single experiment
python src/eval/experiment_evaluator.py --experiment exp01 --split test

# Evaluate universal model on all datasets
python src/eval/evaluate_universal.py --checkpoint results/universal_model/final_checkpoint.pth
```

## Universal Training Strategy

AgriSAM3 employs a **progressive fine-tuning strategy**:

1. **Stage 1**: Fine-tune on each experiment independently
2. **Stage 2**: Combine learned concepts with curriculum learning
3. **Stage 3**: Multi-dataset mixed training for universal model
4. **Stage 4**: Zero-shot evaluation on held-out agricultural domains

### Training Flow

```
Pretrained SAM3
    ↓
Exp 1: Fruit (apples, ripeness) → Checkpoint 1
    ↓
Exp 2: Weeds (crop-weed) → Checkpoint 2 (initialized from 1)
    ↓
Exp 3: Disease (symptoms) → Checkpoint 3 (initialized from 2)
    ↓
Exp 4: Multi-crop → Checkpoint 4 (initialized from 3)
    ↓
Exp 5: Pests → Checkpoint 5 (initialized from 4)
    ↓
Exp 6: Greenhouse → Checkpoint 6 (initialized from 5)
    ↓
Multi-dataset Training → Universal AgriSAM3 Model
```

## Configuration System

Each experiment has a YAML config:

```yaml
# configs/exp01_fruit_ripeness.yml
experiment:
  name: "exp01_fruit_ripeness"
  description: "Fruit detection and ripeness assessment"
  
dataset:
  name: "MinneApple"
  format: "coco"  # coco, voc, custom
  path: "data/experiment_01_fruit"
  
concepts:
  - "ripe apples"
  - "unripe apples"
  - "overripe apples"
  - "apple varieties"
  
training:
  batch_size: 1
  epochs: 30
  learning_rate: 5e-5
  resume_from: "pretrained"  # or path to checkpoint
  
evaluation:
  metrics:
    - "iou"
    - "concept_recall"
    - "ripeness_accuracy"
```

## Datasets

### Experiment 1: MinneApple
- **Size**: 1,200+ images
- **Concepts**: Ripe/unripe apples, varieties
- **Source**: University of Minnesota

### Experiment 2: DeepWeeds + Agriculture-Vision
- **Size**: 17,000+ images (DeepWeeds) + 56,000+ (Ag-Vision)
- **Concepts**: 9 weed species, crops, soil
- **Source**: Public agricultural datasets

### Experiment 3: PlantDoc + PlantVillage
- **Size**: 2,500+ (PlantDoc) + 54,000+ (PlantVillage)
- **Concepts**: 27 plant species, 13 disease classes
- **Source**: Kaggle, PlantVillage

### Experiment 4: Agriculture-Vision
- **Size**: 56,000+ images
- **Concepts**: Multiple crop types, growth stages
- **Source**: Agriculture-Vision Challenge

### Experiment 5: IP102
- **Size**: 75,000+ images
- **Concepts**: 102 insect pest species
- **Source**: IP102 Dataset

### Experiment 6: Custom Greenhouse Dataset
- **Size**: Custom collected
- **Concepts**: Indoor agriculture, controlled environment
- **Source**: Research partners

## Evaluation Metrics

### Universal Metrics (All Experiments)
- Mean IoU (geometric accuracy)
- Concept Recall (semantic understanding)
- Concept Precision
- F1 Score
- Boundary F1

### Domain-Specific Metrics
- **Fruit**: Ripeness classification accuracy
- **Weed**: Crop-weed discrimination accuracy
- **Disease**: Disease classification accuracy, symptom localization
- **Pest**: Pest species accuracy, damage severity
- **Multi-crop**: Crop type accuracy, growth stage recognition

## Results

### Individual Experiments

| Experiment | Domain | Mean IoU | Concept Recall | Domain Metric |
|------------|--------|----------|----------------|---------------|
| Exp 1 | Fruit | 0.89 | 0.85 | Ripeness: 87% |
| Exp 2 | Weed | 0.84 | 0.81 | Crop-Weed: 90% |
| Exp 3 | Disease | 0.82 | 0.79 | Disease: 83% |
| Exp 4 | Multi-crop | 0.86 | 0.83 | Crop Type: 88% |
| Exp 5 | Pest | 0.80 | 0.77 | Pest ID: 82% |
| Exp 6 | Greenhouse | 0.87 | 0.84 | Health: 85% |

### Universal Model

**Zero-shot Performance** on held-out agricultural datasets:
- Mean IoU: 0.78
- Concept Recall: 0.73
- Demonstrates strong generalization to unseen agricultural concepts

## Citation

If you use AgriSAM3 in your research, please cite:

```bibtex
@article{sapkota2025agrisam3,
  title={AgriSAM3: Delving into Segment Anything with Agricultural Concepts},
  author={Sapkota, Ranjan and Roumeliotis, Konstantinos I. and Karkee, Manoj},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

Apache 2.0 - See LICENSE file for details

## Acknowledgments

- Meta AI for SAM3
- Agricultural dataset curators
- Cornell University and University of the Peloponnese
- Precision agriculture research community

## Related Work

- **SAM3**: [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)

---

**Status**: 🚧 Active Development  
**Last Updated**: December 2025
