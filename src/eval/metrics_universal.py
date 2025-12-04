"""
Universal Metrics for Agricultural Segmentation

Comprehensive metrics for evaluating segmentation performance across
all agricultural domains. Includes standard segmentation metrics and
agricultural-specific evaluation measures.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import warnings


class SegmentationMetrics:
    """
    Standard segmentation metrics: IoU, Dice, Precision, Recall, F1
    
    Handles both binary and multi-class segmentation.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in metric computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        self.tn = 0  # True negatives
        
        self.per_class_tp = np.zeros(self.num_classes)
        self.per_class_fp = np.zeros(self.num_classes)
        self.per_class_fn = np.zeros(self.num_classes)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
        """
        Update metrics with new predictions
        
        Args:
            predictions: Predicted masks (H, W) or (B, H, W)
            targets: Ground truth masks (same shape)
            threshold: Binarization threshold for predictions
        """
        # Ensure numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Binarize predictions
        pred_binary = (predictions > threshold).astype(np.uint8)
        target_binary = targets.astype(np.uint8)
        
        # Flatten
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # Remove ignored indices
        if self.ignore_index is not None:
            valid_mask = target_flat != self.ignore_index
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
        
        # Compute confusion matrix elements
        self.tp += np.sum((pred_flat == 1) & (target_flat == 1))
        self.fp += np.sum((pred_flat == 1) & (target_flat == 0))
        self.fn += np.sum((pred_flat == 0) & (target_flat == 1))
        self.tn += np.sum((pred_flat == 0) & (target_flat == 0))
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary with IoU, Dice, Precision, Recall, F1, Accuracy
        """
        epsilon = 1e-7
        
        # IoU (Intersection over Union)
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        iou = (intersection + epsilon) / (union + epsilon)
        
        # Dice coefficient
        dice = (2 * self.tp + epsilon) / (2 * self.tp + self.fp + self.fn + epsilon)
        
        # Precision
        precision = (self.tp + epsilon) / (self.tp + self.fp + epsilon)
        
        # Recall (Sensitivity)
        recall = (self.tp + epsilon) / (self.tp + self.fn + epsilon)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # Accuracy
        accuracy = (self.tp + self.tn + epsilon) / (self.tp + self.tn + self.fp + self.fn + epsilon)
        
        # Specificity
        specificity = (self.tn + epsilon) / (self.tn + self.fp + epsilon)
        
        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity)
        }


class ConceptRecognitionMetrics:
    """
    Metrics for agricultural concept recognition (ripeness, health, disease, etc.)
    
    Evaluates how well the model understands and segments agricultural concepts.
    """
    
    def __init__(self, concepts: List[str]):
        """
        Initialize concept metrics
        
        Args:
            concepts: List of agricultural concepts to evaluate
        """
        self.concepts = concepts
        self.concept_predictions = {c: [] for c in concepts}
        self.concept_targets = {c: [] for c in concepts}
    
    def update(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """
        Update concept metrics
        
        Args:
            predictions: Dictionary mapping concept names to predicted values
            targets: Dictionary mapping concept names to ground truth values
        """
        for concept in self.concepts:
            if concept in predictions and concept in targets:
                self.concept_predictions[concept].extend(predictions[concept].tolist())
                self.concept_targets[concept].extend(targets[concept].tolist())
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute concept recognition metrics
        
        Returns:
            Dictionary mapping concepts to their metrics
        """
        results = {}
        
        for concept in self.concepts:
            if not self.concept_predictions[concept]:
                continue
            
            preds = np.array(self.concept_predictions[concept])
            targets = np.array(self.concept_targets[concept])
            
            # Accuracy
            accuracy = np.mean(preds == targets)
            
            # Per-class metrics
            unique_classes = np.unique(targets)
            per_class_recall = {}
            per_class_precision = {}
            
            for cls in unique_classes:
                cls_mask = targets == cls
                if cls_mask.sum() > 0:
                    recall = np.mean(preds[cls_mask] == cls)
                    per_class_recall[int(cls)] = float(recall)
                
                pred_cls_mask = preds == cls
                if pred_cls_mask.sum() > 0:
                    precision = np.mean(targets[pred_cls_mask] == cls)
                    per_class_precision[int(cls)] = float(precision)
            
            results[concept] = {
                'accuracy': float(accuracy),
                'per_class_recall': per_class_recall,
                'per_class_precision': per_class_precision
            }
        
        return results
    
    def reset(self):
        """Reset accumulated predictions"""
        self.concept_predictions = {c: [] for c in self.concepts}
        self.concept_targets = {c: [] for c in self.concepts}


class BoundaryAccuracyMetrics:
    """
    Metrics for boundary detection quality
    
    Measures how well predicted boundaries match ground truth.
    """
    
    def __init__(self, tolerance_pixels: int = 5):
        """
        Initialize boundary metrics
        
        Args:
            tolerance_pixels: Distance tolerance for boundary matching
        """
        self.tolerance = tolerance_pixels
        self.boundary_scores = []
    
    def update(self, pred_mask: np.ndarray, target_mask: np.ndarray):
        """
        Update boundary metrics
        
        Args:
            pred_mask: Predicted segmentation mask
            target_mask: Ground truth segmentation mask
        """
        try:
            import cv2
            
            # Extract boundaries
            pred_boundaries = self._extract_boundaries(pred_mask)
            target_boundaries = self._extract_boundaries(target_mask)
            
            # Compute boundary F1 score
            score = self._boundary_f1(pred_boundaries, target_boundaries)
            self.boundary_scores.append(score)
        
        except ImportError:
            warnings.warn("OpenCV not available, skipping boundary metrics")
    
    def _extract_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundaries from segmentation mask"""
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundaries = mask.astype(np.uint8) - eroded
        return boundaries
    
    def _boundary_f1(self, pred_boundaries: np.ndarray, target_boundaries: np.ndarray) -> float:
        """Compute F1 score for boundary pixels within tolerance"""
        import cv2
        from scipy.ndimage import distance_transform_edt
        
        # Distance transform
        target_dist = distance_transform_edt(1 - target_boundaries)
        pred_dist = distance_transform_edt(1 - pred_boundaries)
        
        # True positives: predicted boundaries within tolerance of target
        tp = np.sum((pred_boundaries > 0) & (target_dist <= self.tolerance))
        
        # False positives
        fp = np.sum((pred_boundaries > 0) & (target_dist > self.tolerance))
        
        # False negatives
        fn = np.sum((target_boundaries > 0) & (pred_dist > self.tolerance))
        
        # F1 score
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return f1
    
    def compute(self) -> Dict[str, float]:
        """Compute average boundary metrics"""
        if not self.boundary_scores:
            return {'boundary_f1': 0.0}
        
        return {
            'boundary_f1': float(np.mean(self.boundary_scores)),
            'boundary_f1_std': float(np.std(self.boundary_scores))
        }
    
    def reset(self):
        """Reset boundary scores"""
        self.boundary_scores = []


class MultiScaleMetrics:
    """
    Multi-scale evaluation metrics
    
    Evaluates performance across different object sizes (small, medium, large).
    Important for agricultural applications with varying scales.
    """
    
    def __init__(self, small_threshold: int = 32**2, large_threshold: int = 96**2):
        """
        Initialize multi-scale metrics
        
        Args:
            small_threshold: Area threshold for small objects (pixels²)
            large_threshold: Area threshold for large objects (pixels²)
        """
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        
        self.metrics = {
            'small': SegmentationMetrics(),
            'medium': SegmentationMetrics(),
            'large': SegmentationMetrics()
        }
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, 
               instance_masks: List[np.ndarray]):
        """
        Update metrics categorized by object size
        
        Args:
            predictions: Predicted masks
            targets: Ground truth masks
            instance_masks: List of individual instance masks for size computation
        """
        for inst_mask in instance_masks:
            area = np.sum(inst_mask > 0)
            
            if area < self.small_threshold:
                category = 'small'
            elif area < self.large_threshold:
                category = 'medium'
            else:
                category = 'large'
            
            # Update category metrics
            self.metrics[category].update(
                predictions * inst_mask,
                targets * inst_mask
            )
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each scale"""
        return {
            scale: metrics.compute()
            for scale, metrics in self.metrics.items()
        }
    
    def reset(self):
        """Reset all scale metrics"""
        for metrics in self.metrics.values():
            metrics.reset()


class AgriculturalMetricsAggregator:
    """
    Aggregates all agricultural metrics for comprehensive evaluation
    
    Combines segmentation, concept recognition, boundary, and scale metrics.
    """
    
    def __init__(self, concepts: List[str] = None, enable_boundary: bool = True,
                 enable_multiscale: bool = True):
        """
        Initialize metrics aggregator
        
        Args:
            concepts: Agricultural concepts to evaluate
            enable_boundary: Whether to compute boundary metrics
            enable_multiscale: Whether to compute multi-scale metrics
        """
        self.segmentation_metrics = SegmentationMetrics()
        
        self.concept_metrics = None
        if concepts:
            self.concept_metrics = ConceptRecognitionMetrics(concepts)
        
        self.boundary_metrics = BoundaryAccuracyMetrics() if enable_boundary else None
        self.multiscale_metrics = MultiScaleMetrics() if enable_multiscale else None
    
    def update(self, predictions: Dict, targets: Dict):
        """
        Update all metrics
        
        Args:
            predictions: Dictionary with predictions (masks, concepts, etc.)
            targets: Dictionary with ground truth
        """
        # Update segmentation metrics
        if 'masks' in predictions and 'masks' in targets:
            self.segmentation_metrics.update(predictions['masks'], targets['masks'])
        
        # Update concept metrics
        if self.concept_metrics and 'concepts' in predictions and 'concepts' in targets:
            self.concept_metrics.update(predictions['concepts'], targets['concepts'])
        
        # Update boundary metrics
        if self.boundary_metrics and 'masks' in predictions and 'masks' in targets:
            for pred, target in zip(predictions['masks'], targets['masks']):
                self.boundary_metrics.update(pred, target)
        
        # Update multi-scale metrics
        if (self.multiscale_metrics and 'masks' in predictions and 
            'masks' in targets and 'instance_masks' in targets):
            self.multiscale_metrics.update(
                predictions['masks'],
                targets['masks'],
                targets['instance_masks']
            )
    
    def compute(self) -> Dict:
        """Compute all metrics"""
        results = {}
        
        # Segmentation metrics
        results['segmentation'] = self.segmentation_metrics.compute()
        
        # Concept metrics
        if self.concept_metrics:
            results['concepts'] = self.concept_metrics.compute()
        
        # Boundary metrics
        if self.boundary_metrics:
            results['boundary'] = self.boundary_metrics.compute()
        
        # Multi-scale metrics
        if self.multiscale_metrics:
            results['multiscale'] = self.multiscale_metrics.compute()
        
        return results
    
    def reset(self):
        """Reset all metrics"""
        self.segmentation_metrics.reset()
        if self.concept_metrics:
            self.concept_metrics.reset()
        if self.boundary_metrics:
            self.boundary_metrics.reset()
        if self.multiscale_metrics:
            self.multiscale_metrics.reset()


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray,
                             num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix for classification
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth class labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def classification_metrics(predictions: np.ndarray, targets: np.ndarray,
                          class_names: List[str] = None) -> Dict:
    """
    Compute classification metrics with sklearn
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        class_names: Optional class names for report
    
    Returns:
        Dictionary with classification report
    """
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    return report
