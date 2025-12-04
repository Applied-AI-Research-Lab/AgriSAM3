"""
Universal Agricultural Dataset Loaders

Flexible data loading system that handles multiple annotation formats:
- COCO format (most agricultural datasets)
- Pascal VOC format (XML annotations)
- Custom JSON format (roboflow, cvat exports)

Automatically detects format and provides unified interface for training.
Generates agricultural text prompts from annotations and metadata.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
    COCO_AVAILABLE = True
except ImportError:
    warnings.warn("pycocotools not installed. COCO format support limited.")
    COCO_AVAILABLE = False

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AgriculturalDatasetLoader:
    """
    Universal dataset loader for agricultural segmentation tasks
    
    Automatically detects annotation format and creates appropriate dataset.
    Supports COCO, Pascal VOC, and custom JSON formats.
    """
    
    def __init__(self, data_config: Dict, prompt_generator: Any, resolution: int = 1008):
        """
        Initialize universal data loader
        
        Args:
            data_config: Dataset configuration from YAML
            prompt_generator: Agricultural prompt generator instance
            resolution: Image resolution for training (default 1008 for SAM3)
        """
        self.data_config = data_config
        self.prompt_generator = prompt_generator
        self.resolution = resolution
        
        # Detect annotation format
        self.format = self._detect_format()
        print(f"Detected annotation format: {self.format}")
    
    def _detect_format(self) -> str:
        """Auto-detect annotation format from config or file structure"""
        explicit_format = self.data_config.get('format', '').lower()
        if explicit_format in ['coco', 'voc', 'json']:
            return explicit_format
        
        # Try to infer from paths
        annotation_path = self.data_config.get('annotations', {}).get('train', '')
        if annotation_path.endswith('.json'):
            return 'coco'
        elif annotation_path.endswith('.xml') or 'Annotations' in annotation_path:
            return 'voc'
        else:
            return 'json'
    
    def get_dataset(self, split: str) -> Dataset:
        """
        Get dataset for specified split
        
        Args:
            split: One of 'train', 'val', 'test'
        
        Returns:
            Dataset instance for the split
        """
        if self.format == 'coco':
            return COCOAgriculturalDataset(
                self.data_config,
                split,
                self.prompt_generator,
                self.resolution
            )
        elif self.format == 'voc':
            return VOCAgriculturalDataset(
                self.data_config,
                split,
                self.prompt_generator,
                self.resolution
            )
        else:
            return CustomJSONAgriculturalDataset(
                self.data_config,
                split,
                self.prompt_generator,
                self.resolution
            )


class BaseAgriculturalDataset(Dataset):
    """Base class for agricultural datasets with common functionality"""
    
    def __init__(self, data_config: Dict, split: str, 
                 prompt_generator: Any, resolution: int):
        """
        Initialize base dataset
        
        Args:
            data_config: Dataset configuration
            split: Data split (train/val/test)
            prompt_generator: Agricultural prompt generator
            resolution: Target image resolution
        """
        self.data_config = data_config
        self.split = split
        self.prompt_generator = prompt_generator
        self.resolution = resolution
        
        # Augmentation for training
        self.augmentation = self._get_augmentation() if split == 'train' else None
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_augmentation(self) -> A.Compose:
        """Get augmentation pipeline for training"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
        ], additional_targets={'mask': 'mask'})
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (PIL image, numpy array)
        """
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        return image, image_array
    
    def apply_augmentation(self, image_array: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to image and mask"""
        if self.augmentation:
            augmented = self.augmentation(image=image_array, mask=mask)
            return augmented['image'], augmented['mask']
        return image_array, mask
    
    def prepare_masks(self, annotations: List[Dict]) -> np.ndarray:
        """
        Prepare instance segmentation masks from annotations
        
        Args:
            annotations: List of annotation dictionaries
        
        Returns:
            Binary mask array of shape (H, W, N) where N is number of instances
        """
        raise NotImplementedError("Subclasses must implement prepare_masks")


class COCOAgriculturalDataset(BaseAgriculturalDataset):
    """
    Dataset loader for COCO format agricultural datasets
    
    Supports standard COCO JSON with extensions for agricultural attributes:
    - Ripeness level (green, ripe, overripe)
    - Health status (healthy, diseased, pest-damaged)
    - Growth stage (seedling, vegetative, flowering, fruiting)
    - Species/variety information
    """
    
    def __init__(self, data_config: Dict, split: str, 
                 prompt_generator: Any, resolution: int):
        super().__init__(data_config, split, prompt_generator, resolution)
        
        if not COCO_AVAILABLE:
            raise RuntimeError("pycocotools required for COCO format")
        
        # Load COCO annotations
        annotation_path = data_config['annotations'][split]
        self.coco = COCO(annotation_path)
        
        # Get all image IDs with annotations
        self.image_ids = sorted(self.coco.getImgIds())
        
        # Image root directory
        self.image_root = Path(data_config['images'][split])
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item
        
        Returns dictionary with:
        - image: PIL Image
        - image_array: Preprocessed numpy array
        - masks: Binary masks (H, W, N)
        - text_prompts: List of agricultural text prompts
        - attributes: Agricultural attributes per instance
        - metadata: Additional information
        """
        image_id = self.image_ids[idx]
        
        # Load image info
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_root / img_info['file_name']
        
        image, image_array = self.load_and_preprocess_image(str(image_path))
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Prepare masks
        masks = self.prepare_masks(annotations, img_info['height'], img_info['width'])
        
        # Apply augmentation
        if self.split == 'train' and self.augmentation:
            image_array, masks = self.apply_augmentation(image_array, masks)
        
        # Generate text prompts from annotations
        text_prompts = []
        attributes = []
        
        for ann in annotations:
            # Get category
            cat_id = ann['category_id']
            category = self.coco.loadCats(cat_id)[0]['name']
            
            # Extract agricultural attributes from annotation
            agri_attributes = self._extract_attributes(ann)
            attributes.append(agri_attributes)
            
            # Generate text prompt
            prompt = self.prompt_generator.generate_prompt(
                category=category,
                attributes=agri_attributes
            )
            text_prompts.append(prompt)
        
        # Resize image and masks
        image_tensor = self.transform(Image.fromarray(image_array.astype('uint8')))
        masks_tensor = torch.from_numpy(masks).float()
        
        # Resize masks
        if masks.shape[0] > 0:
            masks_resized = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode='nearest'
            ).squeeze(0)
        else:
            masks_resized = torch.zeros((0, self.resolution, self.resolution))
        
        return {
            'image': image,
            'image_array': image_tensor,
            'masks': masks_resized,
            'text_prompts': text_prompts,
            'attributes': attributes,
            'metadata': {
                'image_id': image_id,
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'num_instances': len(annotations)
            }
        }
    
    def prepare_masks(self, annotations: List[Dict], height: int, width: int) -> np.ndarray:
        """Convert COCO annotations to binary masks"""
        if len(annotations) == 0:
            return np.zeros((0, height, width), dtype=np.uint8)
        
        masks = []
        for ann in annotations:
            if 'segmentation' in ann:
                # Polygon or RLE format
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                    mask = coco_mask.decode(rle)
                    if len(mask.shape) == 3:
                        mask = mask.max(axis=2)
                else:
                    # RLE format
                    mask = coco_mask.decode(ann['segmentation'])
                
                masks.append(mask)
        
        if len(masks) > 0:
            return np.stack(masks, axis=0)
        return np.zeros((0, height, width), dtype=np.uint8)
    
    def _extract_attributes(self, annotation: Dict) -> Dict:
        """Extract agricultural attributes from COCO annotation"""
        attributes = {}
        
        # Standard attributes
        if 'ripeness' in annotation:
            attributes['ripeness'] = annotation['ripeness']
        if 'health' in annotation:
            attributes['health'] = annotation['health']
        if 'growth_stage' in annotation:
            attributes['growth_stage'] = annotation['growth_stage']
        if 'variety' in annotation:
            attributes['variety'] = annotation['variety']
        
        # Damage assessment
        if 'damage_type' in annotation:
            attributes['damage_type'] = annotation['damage_type']
        if 'damage_severity' in annotation:
            attributes['damage_severity'] = annotation['damage_severity']
        
        return attributes


class VOCAgriculturalDataset(BaseAgriculturalDataset):
    """
    Dataset loader for Pascal VOC format agricultural datasets
    
    Reads XML annotations with bounding boxes and optional segmentation masks.
    """
    
    def __init__(self, data_config: Dict, split: str, 
                 prompt_generator: Any, resolution: int):
        super().__init__(data_config, split, prompt_generator, resolution)
        
        # Get image and annotation directories
        self.image_dir = Path(data_config['images'][split])
        self.annotation_dir = Path(data_config['annotations'][split])
        
        # Get list of images
        split_file = data_config.get('split_files', {}).get(split)
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_names = [line.strip() for line in f.readlines()]
        else:
            self.image_names = [f.stem for f in self.image_dir.glob('*.jpg')]
        
        print(f"Loaded {len(self.image_names)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item"""
        image_name = self.image_names[idx]
        
        # Load image
        image_path = self.image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_name}.png"
        
        image, image_array = self.load_and_preprocess_image(str(image_path))
        
        # Parse XML annotation
        xml_path = self.annotation_dir / f"{image_name}.xml"
        annotations = self._parse_voc_xml(xml_path)
        
        # Prepare masks from bounding boxes
        height, width = image_array.shape[:2]
        masks = self._boxes_to_masks(annotations, height, width)
        
        # Apply augmentation
        if self.split == 'train' and self.augmentation:
            image_array, masks = self.apply_augmentation(image_array, masks)
        
        # Generate text prompts
        text_prompts = []
        attributes = []
        
        for ann in annotations:
            attributes.append(ann.get('attributes', {}))
            prompt = self.prompt_generator.generate_prompt(
                category=ann['name'],
                attributes=ann.get('attributes', {})
            )
            text_prompts.append(prompt)
        
        # Transform
        image_tensor = self.transform(Image.fromarray(image_array.astype('uint8')))
        masks_tensor = torch.from_numpy(masks).float()
        
        # Resize masks
        if masks.shape[0] > 0:
            masks_resized = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode='nearest'
            ).squeeze(0)
        else:
            masks_resized = torch.zeros((0, self.resolution, self.resolution))
        
        return {
            'image': image,
            'image_array': image_tensor,
            'masks': masks_resized,
            'text_prompts': text_prompts,
            'attributes': attributes,
            'metadata': {
                'image_name': image_name,
                'height': height,
                'width': width,
                'num_instances': len(annotations)
            }
        }
    
    def _parse_voc_xml(self, xml_path: Path) -> List[Dict]:
        """Parse Pascal VOC XML annotation"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            ann = {
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax],
                'attributes': {}
            }
            
            # Extract custom agricultural attributes if present
            attributes = obj.find('attributes')
            if attributes is not None:
                for attr in attributes:
                    ann['attributes'][attr.tag] = attr.text
            
            annotations.append(ann)
        
        return annotations
    
    def _boxes_to_masks(self, annotations: List[Dict], height: int, width: int) -> np.ndarray:
        """Convert bounding boxes to binary masks"""
        if len(annotations) == 0:
            return np.zeros((0, height, width), dtype=np.uint8)
        
        masks = []
        for ann in annotations:
            mask = np.zeros((height, width), dtype=np.uint8)
            xmin, ymin, xmax, ymax = ann['bbox']
            mask[ymin:ymax, xmin:xmax] = 1
            masks.append(mask)
        
        return np.stack(masks, axis=0)
    
    def prepare_masks(self, annotations: List[Dict]) -> np.ndarray:
        """Implemented in __getitem__"""
        pass


class CustomJSONAgriculturalDataset(BaseAgriculturalDataset):
    """
    Dataset loader for custom JSON format agricultural datasets
    
    Supports formats from roboflow, cvat, labelbox, etc.
    """
    
    def __init__(self, data_config: Dict, split: str, 
                 prompt_generator: Any, resolution: int):
        super().__init__(data_config, split, prompt_generator, resolution)
        
        # Load JSON annotations
        annotation_path = data_config['annotations'][split]
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_root = Path(data_config['images'][split])
        
        print(f"Loaded {len(self.annotations)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item"""
        item = self.annotations[idx]
        
        # Load image
        image_path = self.image_root / item['image_path']
        image, image_array = self.load_and_preprocess_image(str(image_path))
        
        # Parse annotations
        height, width = image_array.shape[:2]
        masks = self._parse_custom_annotations(item['annotations'], height, width)
        
        # Apply augmentation
        if self.split == 'train' and self.augmentation:
            image_array, masks = self.apply_augmentation(image_array, masks)
        
        # Generate prompts
        text_prompts = []
        attributes = []
        
        for ann in item['annotations']:
            attributes.append(ann.get('attributes', {}))
            prompt = self.prompt_generator.generate_prompt(
                category=ann.get('label', 'object'),
                attributes=ann.get('attributes', {})
            )
            text_prompts.append(prompt)
        
        # Transform
        image_tensor = self.transform(Image.fromarray(image_array.astype('uint8')))
        masks_tensor = torch.from_numpy(masks).float()
        
        # Resize masks
        if masks.shape[0] > 0:
            masks_resized = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode='nearest'
            ).squeeze(0)
        else:
            masks_resized = torch.zeros((0, self.resolution, self.resolution))
        
        return {
            'image': image,
            'image_array': image_tensor,
            'masks': masks_resized,
            'text_prompts': text_prompts,
            'attributes': attributes,
            'metadata': {
                'image_path': item['image_path'],
                'height': height,
                'width': width,
                'num_instances': len(item['annotations'])
            }
        }
    
    def _parse_custom_annotations(self, annotations: List[Dict], 
                                  height: int, width: int) -> np.ndarray:
        """Parse custom annotation format to masks"""
        if len(annotations) == 0:
            return np.zeros((0, height, width), dtype=np.uint8)
        
        masks = []
        for ann in annotations:
            if 'polygon' in ann:
                mask = self._polygon_to_mask(ann['polygon'], height, width)
            elif 'bbox' in ann:
                mask = self._bbox_to_mask(ann['bbox'], height, width)
            else:
                continue
            masks.append(mask)
        
        if len(masks) > 0:
            return np.stack(masks, axis=0)
        return np.zeros((0, height, width), dtype=np.uint8)
    
    def _polygon_to_mask(self, polygon: List[List[float]], 
                        height: int, width: int) -> np.ndarray:
        """Convert polygon to binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask
    
    def _bbox_to_mask(self, bbox: List[float], height: int, width: int) -> np.ndarray:
        """Convert bounding box to binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = bbox
        mask[int(y):int(y+h), int(x):int(x+w)] = 1
        return mask
    
    def prepare_masks(self, annotations: List[Dict]) -> np.ndarray:
        """Implemented in __getitem__"""
        pass
