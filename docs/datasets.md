# Agricultural Datasets for AgriSAM3

Comprehensive guide to datasets used in the AgriSAM3 project.

## Overview

AgriSAM3 uses 6 diverse agricultural datasets covering different domains:

| Experiment | Dataset | Domain | Images | Classes | Task |
|------------|---------|--------|--------|---------|------|
| Exp01 | MinneApple | Fruit ripeness | 1,200 | 3 | Instance segmentation + ripeness |
| Exp02 | DeepWeeds + AgriVision | Crop-weed | 18,000 | 10 | Semantic segmentation |
| Exp03 | PlantDoc + PlantVillage | Disease | 20,000 | 13 | Disease segmentation |
| Exp04 | Agriculture-Vision | Multi-crop (aerial) | 15,000 | 9 | Field segmentation |
| Exp05 | IP102 | Insect pests | 25,000 | 102 | Pest detection |
| Exp06 | Custom Greenhouse | Indoor agriculture | 8,000 | 10 | Plant + infrastructure |

---

## Experiment 01: MinneApple Dataset

**Domain:** Fruit Detection and Ripeness Assessment

### Description
MinneApple is a benchmark dataset for apple detection, segmentation, and counting in orchard environments. Extended with ripeness annotations for this project.

### Statistics
- **Total images:** 1,200
- **Annotations:** 18,000+ apple instances
- **Image resolution:** Varies (typically 1080p)
- **Environment:** Outdoor orchards, Minnesota
- **Ripeness classes:** Unripe (green), Ripe (red/yellow), Overripe

### Download Instructions

```bash
# Create directory
mkdir -p data/minneapple
cd data/minneapple

# Option 1: Official GitHub
git clone https://github.com/nicolaihaeni/MinneApple.git
cd MinneApple
# Follow their data preparation instructions

# Option 2: Direct download
wget https://conservancy.umn.edu/bitstream/handle/11299/206575/MinneApple.zip
unzip MinneApple.zip
```

### Data Structure

```
data/minneapple/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── annotations.json (COCO format)
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

### Annotation Format

COCO JSON with custom attributes:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234,
      "bbox": [x, y, width, height],
      "ripeness": "ripe",           // Custom: unripe, ripe, overripe
      "color": "red",                // Custom: green, yellow, red
      "variety": "honeycrisp"        // Custom: apple variety
    }
  ],
  "categories": [
    {"id": 1, "name": "apple_unripe"},
    {"id": 2, "name": "apple_ripe"},
    {"id": 3, "name": "apple_overripe"}
  ]
}
```

### Citation

```bibtex
@article{haeni2020minneapple,
  title={MinneApple: A Benchmark Dataset for Apple Detection and Segmentation},
  author={Haeni, Nicolai and Roy, Pravakar and Isler, Volkan},
  journal={IEEE Robotics and Automation Letters},
  year={2020}
}
```

---

## Experiment 02: DeepWeeds + Agriculture-Vision

**Domain:** Crop-Weed Segmentation

### DeepWeeds Dataset

**Description:** Images of Australian native and invasive weed species.

#### Statistics
- **Total images:** 17,509
- **Weed species:** 8 native + invasive
- **Image resolution:** 256×256
- **Environment:** Rangeland, Queensland, Australia

#### Download

```bash
mkdir -p data/crop_weed/deepweeds
cd data/crop_weed/deepweeds

# Kaggle dataset
kaggle datasets download -d imsparsh/deepweeds
unzip deepweeds.zip

# Or manual download from:
# https://github.com/AlexOlsen/DeepWeeds
```

#### Classes
1. Chinee Apple
2. Lantana
3. Parkinsonia
4. Parthenium
5. Prickly Acacia
6. Rubber Vine
7. Siam Weed
8. Snake Weed
9. Negative (no weed)

### Agriculture-Vision Dataset

**Description:** Aerial farmland images for pattern recognition.

#### Statistics
- **Total images:** 56,944 aerial images
- **Coverage:** 3,432 farmland scenes
- **Resolution:** 512×512 patches
- **Location:** US farmland

#### Download

```bash
mkdir -p data/crop_weed/agriculture_vision
cd data/crop_weed/agriculture_vision

# Registration required
# Visit: https://www.agriculture-vision.com/
# Download: https://github.com/SHI-Labs/Agriculture-Vision
```

#### Pattern Classes
- Double plant
- Drydown
- Endrow
- Nutrient deficiency
- Planter skip
- Water
- Waterway
- Weed cluster

### Combined Dataset Structure

```
data/crop_weed/
├── train/
│   ├── images/
│   │   ├── deepweeds_0001.jpg
│   │   ├── agrivision_0001.jpg
│   │   └── ...
│   └── annotations.json
├── val/
└── test/
```

### Citations

```bibtex
@article{olsen2019deepweeds,
  title={DeepWeeds: A multiclass weed species image dataset for deep learning},
  author={Olsen, Alex and others},
  journal={Scientific reports},
  year={2019}
}

@inproceedings{chiu2020agriculture,
  title={Agriculture-vision: A large aerial image database for agricultural pattern analysis},
  author={Chiu, Man-Tsung and others},
  booktitle={CVPR},
  year={2020}
}
```

---

## Experiment 03: PlantDoc + PlantVillage

**Domain:** Plant Disease Detection

### PlantDoc Dataset

**Description:** Indian plant disease dataset covering 13 classes across 17 plant species.

#### Statistics
- **Total images:** 2,598
- **Plant species:** 17 (including crops and fruits)
- **Disease classes:** 13
- **Resolution:** Varies (high quality)

#### Download

```bash
mkdir -p data/plant_disease/plantdoc
cd data/plant_disease/plantdoc

# GitHub repository
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
```

#### Classes
- Apple scab
- Apple black rot
- Apple cedar rust
- Tomato early blight
- Tomato late blight
- Tomato leaf mold
- Grape black rot
- Grape esca
- Corn rust
- Potato early blight
- Healthy leaf

### PlantVillage Dataset

**Description:** Large-scale plant disease classification dataset.

#### Statistics
- **Total images:** 54,305
- **Plant species:** 14
- **Disease classes:** 38 (including healthy)
- **Resolution:** 256×256

#### Download

```bash
mkdir -p data/plant_disease/plantvillage
cd data/plant_disease/plantvillage

# Kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip

# Or GitHub
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
```

#### Major Classes
- Tomato: 10 classes (9 diseases + healthy)
- Potato: 3 classes (2 diseases + healthy)
- Corn: 4 classes (3 diseases + healthy)
- Grape: 4 classes (3 diseases + healthy)
- Apple: 4 classes (3 diseases + healthy)
- Plus: Pepper, Cherry, Peach, Strawberry, etc.

### Combined Structure

```
data/plant_disease/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
└── test/

# Annotation includes disease attributes
{
  "id": 1,
  "disease": "tomato_early_blight",
  "severity": "moderate",    // mild, moderate, severe
  "host_plant": "tomato",
  "health": "diseased"
}
```

### Citations

```bibtex
@article{singh2020plantdoc,
  title={PlantDoc: A Dataset for Visual Plant Disease Detection},
  author={Singh, Davinder and others},
  journal={arXiv preprint arXiv:1911.10317},
  year={2020}
}

@article{hughes2015open,
  title={An open access repository of images on plant health},
  author={Hughes, David and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

---

## Experiment 04: Agriculture-Vision (Aerial Fields)

**Domain:** Multi-Crop Field Segmentation

### Description
Semantic segmentation of agricultural fields from aerial imagery. Focus on field boundaries and crop types.

### Statistics
- **Total images:** 56,944 tiles from 3,432 scenes
- **Resolution:** 512×512 pixels
- **Coverage:** 69,758 km² farmland
- **Location:** United States
- **Sensor:** RGB aerial imagery

### Download

```bash
mkdir -p data/agriculture_vision
cd data/agriculture_vision

# Official website (registration required)
# https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021

# GitHub repository
git clone https://github.com/SHI-Labs/Agriculture-Vision.git
```

### Data Structure

```
data/agriculture_vision/
├── train/
│   ├── images/
│   │   ├── rgb/       # RGB images
│   │   ├── nir/       # Near-infrared (optional)
│   │   └── boundaries/ # Field boundaries
│   └── labels/
│       ├── double_plant/
│       ├── drydown/
│       └── ...
├── val/
└── test/
```

### Annotation Format

Multi-label semantic segmentation:
- Each pattern type has separate binary mask
- Multiple patterns can overlap
- Field boundaries provided separately

### Crop Types
- Corn
- Soybeans
- Wheat
- Cotton
- Rice
- Sugarcane

### Citation

```bibtex
@inproceedings{chiu2020agriculture,
  title={Agriculture-vision: A large aerial image database for agricultural pattern analysis},
  author={Chiu, Man-Tsung and Xu, Xingqian and Wang, Yunchao and others},
  booktitle={CVPR},
  pages={2828--2838},
  year={2020}
}
```

---

## Experiment 05: IP102 (Insect Pests)

**Domain:** Agricultural Pest Detection

### Description
Large-scale insect pest recognition dataset with 102 classes of agricultural pests.

### Statistics
- **Total images:** 75,222
- **Pest classes:** 102
- **Super-classes:** 8 orders (Hemiptera, Lepidoptera, etc.)
- **Resolution:** Varies
- **Environment:** Lab + field conditions

### Download

```bash
mkdir -p data/ip102
cd data/ip102

# GitHub repository
git clone https://github.com/xpwu95/IP102.git

# Kaggle (alternative)
kaggle datasets download -d rtlmhjbn/ip02-dataset
```

### Data Structure

```
data/ip102/
├── train/
│   ├── images/
│   │   ├── 0/    # Class folders
│   │   ├── 1/
│   │   └── ...
│   └── annotations.json
├── val/
└── test/
```

### Major Pest Categories

**Hemiptera (Bugs):**
- Aphids
- Scale insects
- Whiteflies
- Planthoppers

**Lepidoptera (Moths/Butterflies):**
- Armyworms
- Cutworms
- Bollworms
- Leaf miners

**Coleoptera (Beetles):**
- Weevils
- Lady beetles
- Ground beetles

**Orthoptera (Grasshoppers):**
- Locusts
- Crickets

**Others:**
- Thrips
- Mites
- Flies
- Wasps

### Annotation Format

```json
{
  "id": 1,
  "pest_species": "aphid_green_peach",
  "pest_order": "Hemiptera",
  "host_plant": "tomato",
  "damage_type": "leaf_damage",
  "severity": "moderate",
  "bbox": [x, y, w, h],
  "segmentation": [...]
}
```

### Citation

```bibtex
@article{wu2019ip102,
  title={IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition},
  author={Wu, Xiaoping and Zhan, Chi and Lai, Yu-Kun and others},
  journal={CVPR},
  year={2019}
}
```

---

## Experiment 06: Greenhouse Dataset

**Domain:** Indoor Agriculture

### Description
Custom dataset for greenhouse plant segmentation. Combines existing greenhouse datasets and synthetic data.

### Sources

**1. Greenhouse Plant Detection:**
```bash
# Available at Roboflow Universe
# https://universe.roboflow.com/search?q=greenhouse

mkdir -p data/greenhouse
cd data/greenhouse
```

**2. Indoor Plant Datasets:**
- GrowSpace dataset
- Indoor tomato dataset
- Lettuce growth dataset

**3. Synthetic Data:**
- Unreal Engine simulations
- Blender greenhouse scenes

### Statistics
- **Total images:** 8,000
- **Plants:** Tomato, lettuce, cucumber, pepper, strawberry
- **Resolution:** 1920×1080
- **Lighting:** Natural, LED grow lights, mixed

### Data Structure

```
data/greenhouse/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
└── test/

# Annotations include greenhouse-specific attributes
{
  "plant_type": "tomato",
  "growth_stage": "flowering",
  "health": "healthy",
  "lighting_type": "led_grow_light",
  "container_type": "hydroponic",
  "infrastructure": ["grow_light", "support_pole"]
}
```

### Collection Guidelines

If collecting your own data:

1. **Coverage:**
   - Multiple growth stages
   - Different lighting conditions
   - Various plant densities
   - Include infrastructure

2. **Capture:**
   - High resolution (1080p+)
   - Consistent angles
   - Well-lit images
   - Include context

3. **Annotation:**
   - Instance segmentation for plants
   - Semantic segmentation for infrastructure
   - Growth stage labels
   - Health assessment

---

## Data Preprocessing

### Format Conversion

All datasets are converted to COCO format:

```bash
# Convert Pascal VOC to COCO
python scripts/convert_voc_to_coco.py \
    --input_dir data/my_dataset/VOC \
    --output_json data/my_dataset/train/annotations.json

# Convert custom JSON
python scripts/convert_custom_to_coco.py \
    --input_dir data/my_dataset \
    --format roboflow \
    --output_json data/my_dataset/train/annotations.json
```

### Data Validation

```bash
# Validate dataset
python scripts/validate_dataset.py \
    --data_dir data/minneapple \
    --split train

# Visualize annotations
python scripts/visualize_annotations.py \
    --annotations data/minneapple/train/annotations.json \
    --images data/minneapple/train/images \
    --num_samples 10
```

### Data Augmentation

Augmentation is handled automatically by the data loaders. See configs for augmentation parameters.

---

## Dataset Storage Requirements

| Dataset | Raw Size | Processed Size | Total |
|---------|----------|----------------|-------|
| MinneApple | 2 GB | 3 GB | 5 GB |
| DeepWeeds | 1.5 GB | 2 GB | 3.5 GB |
| Agriculture-Vision | 12 GB | 15 GB | 27 GB |
| PlantDoc | 500 MB | 800 MB | 1.3 GB |
| PlantVillage | 3 GB | 4 GB | 7 GB |
| IP102 | 8 GB | 10 GB | 18 GB |
| Greenhouse | 5 GB | 6 GB | 11 GB |
| **Total** | **32 GB** | **40.8 GB** | **~73 GB** |

Plus outputs (~100 GB), recommend **200 GB total storage**.

---

## Ethical Considerations

### Data Usage Rights
- All datasets used are publicly available for research
- Cite original papers when using datasets
- Check licenses for commercial use

### Data Privacy
- No personal information in datasets
- Outdoor/agricultural scenes only
- No sensitive locations

### Fair Use
- Datasets cover diverse geographic regions
- Multiple crop types and conditions
- Balanced representation when possible

---

## Additional Resources

### Dataset Tools
- **COCO API:** Annotation handling
- **Roboflow:** Dataset management
- **CVAT:** Annotation tool
- **LabelMe:** Alternative annotator

### Related Datasets
- **PlantCLEF:** Plant species identification
- **Leaf Counting Challenge:** Plant phenotyping
- **Global Wheat Detection:** Wheat head detection
- **Crop Disease Detection Challenge:** Cassava diseases

---

## Contributing New Datasets

To add your own dataset to AgriSAM3:

1. **Format data in COCO JSON**
2. **Create config file** in `configs/`
3. **Add agricultural attributes** to annotations
4. **Test with validation script**
5. **Document in this file**

See `docs/custom_datasets.md` for detailed instructions.
