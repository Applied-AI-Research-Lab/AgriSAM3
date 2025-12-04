# AgriSAM3: Experimental Methodology

Detailed documentation of the experimental design, rationale, and progressive training strategy.

## Table of Contents
1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Experimental Design](#experimental-design)
4. [Progressive Training Strategy](#progressive-training-strategy)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Expected Results](#expected-results)

---

## Overview

AgriSAM3 explores universal agricultural segmentation through progressive fine-tuning of SAM3 (Segment Anything Model 3) across six diverse agricultural domains. The project aims to build a single model capable of segmenting "anything" in agriculture.

### Core Hypothesis

**Vision-language models like SAM3 can learn universal agricultural concepts through progressive training across diverse domains, enabling generalization to novel agricultural scenarios.**

### Key Innovation

Instead of training domain-specific models, we progressively fine-tune a single model across domains, allowing knowledge transfer and concept composition:

```
Fruit understanding → Vegetation types → Health assessment → 
Scale variation → Small objects → Indoor conditions → Universal model
```

---

## Research Questions

### RQ1: Domain Coverage
**Can a single model effectively segment across all major agricultural domains?**

**Domains tested:**
1. **Fruit production:** Ripeness assessment
2. **Field agriculture:** Crop-weed discrimination
3. **Plant pathology:** Disease detection
4. **Precision agriculture:** Aerial field mapping
5. **Pest management:** Insect identification
6. **Controlled environment:** Greenhouse operations

**Evaluation:** Per-domain IoU and universal model average performance

### RQ2: Knowledge Transfer
**Does progressive training improve performance compared to domain-specific models?**

**Comparison:**
- **Baseline:** Train each domain independently from pretrained SAM3
- **Progressive:** Train sequentially, each building on previous
- **Metric:** Improvement in IoU, data efficiency

**Expected outcome:** Progressive training achieves better performance with less data per domain

### RQ3: Concept Composition
**Can the model combine concepts across domains?**

**Test cases:**
- Ripeness (Exp01) + Disease (Exp03) → Diseased fruit at different ripeness
- Crop type (Exp02) + Health (Exp03) → Unhealthy crop identification
- Pest (Exp05) + Crop (Exp02) → Pest-specific crop damage

**Evaluation:** Zero-shot performance on composite concepts not seen during training

### RQ4: Scale Generalization
**How does the model handle varying scales and perspectives?**

**Scales tested:**
- **Close-up:** Individual fruits, leaves (Exp01, Exp03)
- **Medium:** Plant-level segmentation (Exp02, Exp06)
- **Aerial:** Field-level boundaries (Exp04)
- **Small objects:** Insects (Exp05)

**Metric:** Multi-scale IoU breakdown (small/medium/large objects)

### RQ5: Real-world Applicability
**Does the model work under diverse real-world conditions?**

**Variations:**
- Lighting: Outdoor sun, grow lights, shadows
- Occlusion: Dense vegetation, overlapping plants
- Weather: Different seasons, rain, fog
- Perspective: Ground-level, elevated, aerial

**Evaluation:** Performance on held-out test sets with natural variations

---

## Experimental Design

### 6 Comprehensive Experiments

#### Experiment 01: Fruit Ripeness Segmentation
**Goal:** Establish fruit understanding and ripeness assessment

**Dataset:** MinneApple (1,200 images)

**Concepts:**
- Object detection (apples)
- Ripeness classification (unripe/ripe/overripe)
- Color-based reasoning

**Why first?**
- Well-defined objects with clear boundaries
- Introduces basic agricultural concepts (ripeness, color)
- Foundation for fruit/vegetable understanding

**Expected performance:** IoU 0.75, Ripeness accuracy 0.85

---

#### Experiment 02: Crop-Weed Segmentation
**Goal:** Learn to distinguish crops from weeds

**Datasets:** DeepWeeds + Agriculture-Vision (18,000 images)

**Concepts:**
- Vegetation classification
- Species identification
- Growth pattern recognition

**Why second?**
- Extends from fruit (Exp01) to general vegetation
- Introduces classification between desired/undesired plants
- Critical for precision agriculture

**Progressive benefit:** Fruit understanding transfers to plant structure recognition

**Expected performance:** IoU 0.72, Classification accuracy 0.88

---

#### Experiment 03: Plant Disease Segmentation
**Goal:** Detect and segment diseased plant regions

**Datasets:** PlantDoc + PlantVillage (20,000 images)

**Concepts:**
- Health assessment
- Disease classification
- Symptom severity estimation

**Why third?**
- Adds health dimension to vegetation knowledge
- Requires fine-grained visual reasoning
- Builds on plant structure from Exp01-02

**Progressive benefit:** Understanding healthy plant appearance helps identify anomalies

**Expected performance:** IoU 0.68, Disease accuracy 0.82

---

#### Experiment 04: Multi-Crop Field Segmentation
**Goal:** Segment fields and crop types from aerial imagery

**Dataset:** Agriculture-Vision (15,000 images)

**Concepts:**
- Aerial perspective understanding
- Field boundary detection
- Large-scale crop type mapping

**Why fourth?**
- Tests scale generalization (close-up → aerial)
- Different perspective challenges the model
- Real-world precision agriculture application

**Progressive benefit:** Crop knowledge from ground level extends to aerial view

**Expected performance:** IoU 0.70, Boundary IoU 0.65

---

#### Experiment 05: Insect Pest Segmentation
**Goal:** Detect and identify agricultural pests

**Dataset:** IP102 (25,000 images, 102 pest classes)

**Concepts:**
- Small object detection
- Fine-grained species classification
- Pest-crop interaction

**Why fifth?**
- Introduces small object detection challenge
- Adds pest dimension to agricultural understanding
- Tests fine-grained classification ability

**Progressive benefit:** Crop and disease knowledge helps understand pest damage context

**Expected performance:** IoU 0.62 (small objects), Detection rate 0.85

---

#### Experiment 06: Greenhouse Segmentation
**Goal:** Segment plants and infrastructure in controlled environments

**Dataset:** Custom greenhouse dataset (8,000 images)

**Concepts:**
- Indoor agriculture
- Artificial lighting conditions
- Dense plant arrangements
- Infrastructure detection

**Why sixth (final)?**
- Completes domain coverage (outdoor → indoor)
- Tests under controlled conditions
- Final integration of all previous knowledge

**Progressive benefit:** All previous concepts (ripeness, health, growth stage) apply in greenhouse context

**Expected performance:** IoU 0.73, Health accuracy 0.86

---

## Progressive Training Strategy

### Training Order Rationale

```
Pretrained SAM3 (general segmentation)
    ↓
[Exp01] Basic agricultural concepts: fruits, ripeness, color
    ↓
[Exp02] Vegetation classification: crops vs weeds, species
    ↓
[Exp03] Health assessment: diseases, symptoms, damage
    ↓
[Exp04] Scale variation: aerial view, field boundaries
    ↓
[Exp05] Small objects: pests, fine-grained details
    ↓
[Exp06] Environment adaptation: indoor, controlled conditions
    ↓
Universal Agricultural Model
```

### Knowledge Transfer Mechanisms

**1. Visual Feature Transfer**
- Low-level: Edges, textures, colors
- Mid-level: Plant structures, leaf patterns
- High-level: Agricultural concepts

**2. Text Prompt Learning**
- Agricultural vocabulary building
- Concept composition in language space
- Domain-specific terminology

**3. Concept Accumulation**
Each experiment adds new concepts while retaining previous:
- After Exp01: Knows fruits, ripeness
- After Exp02: + crops, weeds, species
- After Exp03: + health, diseases
- After Exp04: + field boundaries, aerial view
- After Exp05: + pests, damage patterns
- After Exp06: + indoor conditions, infrastructure

### Hyperparameter Schedule

**Learning Rate Decay:**
```
Exp01: 5e-5  (from pretrained)
Exp02: 3e-5  (fine-tuning)
Exp03: 2.5e-5
Exp04: 2e-5
Exp05: 3e-5  (boost for new challenge)
Exp06: 1.5e-5 (final refinement)
```

**Module Freezing:**
- **Tracker:** Always frozen (per SAM3 design)
- **Vision encoder:** Trainable throughout (learns visual features)
- **Text encoder:** Trainable throughout (learns agricultural concepts)
- **Detector:** Always trainable (adapts to domains)

**Loss Weight Evolution:**
- **Segmentation:** Constant 1.0 (always important)
- **Grounding:** 0.5 (vision-language alignment)
- **Domain-specific:** 0.3-0.6 (varies by experiment)

---

## Evaluation Methodology

### Metrics Framework

**1. Standard Segmentation Metrics**
- **IoU (Intersection over Union):** Primary metric
- **Dice coefficient:** Alternative overlap measure
- **Precision:** How many predictions are correct
- **Recall:** How many targets are found
- **F1 score:** Harmonic mean of precision/recall

**2. Domain-Specific Metrics**
- **Ripeness accuracy:** Classification accuracy for ripeness levels
- **Disease recall:** Critical to catch all diseased regions
- **Boundary IoU:** Field boundary quality
- **Small object detection rate:** Pest finding ability

**3. Multi-Scale Metrics**
- Small objects (< 32² pixels): Pest-specific
- Medium objects (32²-96²): Plant-level
- Large objects (> 96²): Field-level

**4. Concept Recognition**
- Accuracy per agricultural concept
- Per-class recall for fine-grained classification
- Confusion matrices for concept understanding

### Evaluation Protocol

**Test Set Validation:**
- Each experiment: Hold-out test set (never seen during training)
- Geographic diversity: Different locations/farms when possible
- Temporal diversity: Different seasons/growth stages

**Universal Model Testing:**
1. **Per-domain evaluation:** Test on each experiment's test set
2. **Cross-domain:** Train on 5 domains, test on 6th
3. **Zero-shot concepts:** Test composite concepts not in training
4. **Real-world validation:** New images from actual farms

**Baselines for Comparison:**
1. **SAM2 (Segment Anything Model 2):** Previous generation
2. **Domain-specific models:** Individual training per experiment
3. **Other segmentation models:** DeepLabV3+, Mask R-CNN

---

## Expected Results

### Per-Experiment Performance Targets

| Experiment | IoU | Domain Metric | Training Time |
|------------|-----|---------------|---------------|
| Exp01: Fruit Ripeness | 0.75 | Ripeness: 0.85 | 4 hours |
| Exp02: Crop-Weed | 0.72 | Classification: 0.88 | 6 hours |
| Exp03: Disease | 0.68 | Disease ID: 0.82 | 7 hours |
| Exp04: Multi-Crop | 0.70 | Boundary: 0.65 | 5 hours |
| Exp05: Pests | 0.62 | Detection: 0.85 | 8 hours |
| Exp06: Greenhouse | 0.73 | Health: 0.86 | 5 hours |

**Total training time:** ~35 hours on A100 GPU

### Universal Model Performance

**Average across domains:**
- Mean IoU: 0.70 ± 0.04
- Best domain: Fruit ripeness (0.75)
- Most challenging: Pests (0.62, small objects)

**Improvement over baselines:**
- vs SAM2: +8-12% IoU (vision-language advantage)
- vs Domain-specific: +5-8% IoU (progressive learning)
- vs Independent training: +3-5% IoU (knowledge transfer)

### Knowledge Transfer Benefits

**Data efficiency:**
- Later experiments require 20-30% less training epochs
- Faster convergence due to agricultural knowledge
- Better generalization to test sets

**Concept composition:**
- Zero-shot: 70-80% of fully-trained performance
- Novel combinations: Successful in 75% of cases
- Vocabulary: Rich agricultural concept space

---

## Validation Strategy

### Statistical Significance
- Multiple runs with different random seeds
- Confidence intervals on metrics
- Significance testing (t-test, p < 0.05)

### Ablation Studies
1. **Progressive vs Independent:** Compare training strategies
2. **Module freezing:** Effect of freezing different components
3. **Loss weights:** Optimal balance for multi-task learning
4. **Data augmentation:** Impact on generalization

### Failure Analysis
- Categorize error types
- Identify challenging scenarios
- Guide future improvements

---

## Broader Impact

### Agricultural Applications
1. **Precision farming:** Targeted intervention
2. **Crop monitoring:** Automated health assessment
3. **Pest management:** Early detection
4. **Yield prediction:** Field analysis
5. **Quality control:** Ripeness assessment
6. **Resource optimization:** Efficient spraying/harvesting

### Scientific Contributions
1. **Universal agricultural AI:** Single model for multiple tasks
2. **Progressive learning:** Knowledge transfer methodology
3. **Vision-language in agriculture:** SAM3 application
4. **Benchmark:** Public dataset and evaluation framework

### Limitations
- Requires labeled data for each domain
- Computationally intensive training
- May not generalize to unseen crop types immediately
- Performance varies with image quality

---

## Future Directions

### Model Improvements
- Multi-modal: Include NIR, thermal imagery
- Temporal: Video understanding for growth tracking
- 3D: Point cloud integration
- Uncertainty: Confidence estimation

### New Domains
- Aquaculture
- Livestock monitoring
- Post-harvest quality
- Agricultural robotics

### Practical Deployment
- Edge deployment: Mobile/embedded systems
- Real-time processing: Video stream analysis
- Active learning: Continuous improvement with user feedback

---

## Reproducibility

All code, configs, and documentation are open-source:
- **Code:** AgriSam3/ directory
- **Configs:** configs/*.yml
- **Scripts:** experiments/*.sh
- **Docs:** docs/*.md

Random seeds fixed for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Exact environment specified in requirements.txt.

---

## Conclusion

AgriSAM3's experimental methodology combines rigorous scientific practice with practical agricultural needs. Through progressive training across diverse domains, we aim to create a universal agricultural segmentation model that advances both AI research and precision agriculture applications.
