"""
Training Utilities for AgriSAM3

Common utilities for training management:
- Logging setup and management
- Checkpoint saving and loading
- Metrics computation
- Visualization helpers
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(output_dir: Path, experiment_name: str) -> logging.Logger:
    """
    Setup logging for training
    
    Args:
        output_dir: Directory for log files
        experiment_name: Experiment identifier
    
    Returns:
        Configured logger
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   best_val_loss: float,
                   training_history: Dict,
                   output_dir: Path,
                   is_best: bool = False):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_val_loss: Best validation loss achieved
        training_history: Training metrics history
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, latest_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)


def load_checkpoint(checkpoint_path: str, 
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> Dict:
    """
    Load checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
    
    Returns:
        Dictionary with checkpoint metadata
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'training_history': checkpoint.get('training_history', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }
    
    print(f"Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"Best validation loss: {metadata['best_val_loss']:.4f}")
    
    return metadata


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute segmentation metrics
    
    Args:
        predictions: Predicted masks (B, H, W) or (B, N, H, W)
        targets: Ground truth masks (same shape)
        threshold: Binarization threshold for predictions
    
    Returns:
        Dictionary with metrics (IoU, Dice, Precision, Recall, F1)
    """
    # Binarize predictions
    pred_binary = (torch.sigmoid(predictions) > threshold).float()
    target_binary = targets.float()
    
    # Flatten for computation
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # IoU (Intersection over Union)
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    # Dice coefficient
    dice = (2 * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
    
    # Precision and Recall
    true_positive = intersection
    false_positive = pred_flat.sum() - intersection
    false_negative = target_flat.sum() - intersection
    
    precision = (true_positive + 1e-7) / (true_positive + false_positive + 1e-7)
    recall = (true_positive + 1e-7) / (true_positive + false_negative + 1e-7)
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    metrics = {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
    
    return metrics


def compute_agricultural_metrics(predictions: Dict, 
                                 targets: Dict,
                                 domain: str) -> Dict[str, float]:
    """
    Compute domain-specific agricultural metrics
    
    Args:
        predictions: Dictionary with predictions including domain-specific outputs
        targets: Dictionary with ground truth
        domain: Agricultural domain (ripeness, disease, etc.)
    
    Returns:
        Dictionary with domain-specific metrics
    """
    metrics = {}
    
    if domain == 'fruit_ripeness':
        # Ripeness classification accuracy
        if 'ripeness_logits' in predictions and 'ripeness_labels' in targets:
            pred_classes = predictions['ripeness_logits'].argmax(dim=1)
            true_classes = targets['ripeness_labels']
            accuracy = (pred_classes == true_classes).float().mean()
            metrics['ripeness_accuracy'] = accuracy.item()
    
    elif domain == 'disease':
        # Disease detection metrics
        if 'disease_logits' in predictions and 'disease_labels' in targets:
            pred_classes = predictions['disease_logits'].argmax(dim=1)
            true_classes = targets['disease_labels']
            accuracy = (pred_classes == true_classes).float().mean()
            metrics['disease_accuracy'] = accuracy.item()
    
    elif domain == 'crop_weed':
        # Crop vs weed classification
        if 'crop_weed_logits' in predictions and 'crop_weed_labels' in targets:
            pred_classes = predictions['crop_weed_logits'].argmax(dim=1)
            true_classes = targets['crop_weed_labels']
            accuracy = (pred_classes == true_classes).float().mean()
            metrics['classification_accuracy'] = accuracy.item()
    
    return metrics


def visualize_predictions(image: np.ndarray,
                        pred_masks: np.ndarray,
                        target_masks: np.ndarray,
                        text_prompts: List[str],
                        save_path: Path):
    """
    Visualize predictions alongside ground truth
    
    Args:
        image: Input image (H, W, 3)
        pred_masks: Predicted masks (N, H, W)
        target_masks: Ground truth masks (N, H, W)
        text_prompts: Text prompts for each instance
        save_path: Where to save visualization
    """
    num_instances = len(pred_masks)
    
    if num_instances == 0:
        return
    
    # Create figure
    fig, axes = plt.subplots(num_instances, 3, figsize=(15, 5 * num_instances))
    if num_instances == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_instances):
        # Original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Image\n{text_prompts[idx]}", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Ground truth mask
        axes[idx, 1].imshow(image)
        axes[idx, 1].imshow(target_masks[idx], alpha=0.5, cmap='Greens')
        axes[idx, 1].set_title("Ground Truth", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Predicted mask
        axes[idx, 2].imshow(image)
        axes[idx, 2].imshow(pred_masks[idx], alpha=0.5, cmap='Reds')
        axes[idx, 2].set_title("Prediction", fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: Dict, save_dir: Path):
    """
    Plot training history curves
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training vs validation loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Loss components
    if 'segmentation_loss' in history and 'grounding_loss' in history:
        axes[0, 1].plot(history['segmentation_loss'], label='Segmentation Loss', linewidth=2)
        axes[0, 1].plot(history['grounding_loss'], label='Grounding Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics
    if 'iou' in history:
        axes[1, 1].plot(history['iou'], label='IoU', linewidth=2)
        if 'dice' in history:
            axes[1, 1].plot(history['dice'], label='Dice', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Segmentation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    
    param_counts = count_parameters(model)
    
    print(f"\nParameter Statistics:")
    print(f"  Total Parameters:     {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    print(f"  Trainable Parameters: {param_counts['trainable']:,} ({param_counts['trainable']/1e6:.1f}M)")
    print(f"  Frozen Parameters:    {param_counts['frozen']:,} ({param_counts['frozen']/1e6:.1f}M)")
    print(f"  Trainable Percentage: {param_counts['trainable_percentage']:.1f}%")
    
    print("\nModule Breakdown:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:30s}: {module_params:12,} params ({module_trainable:12,} trainable)")
    
    print("="*80 + "\n")


def save_experiment_config(config: Dict, output_dir: Path):
    """
    Save experiment configuration
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save config
    """
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved experiment config: {config_path}")


def calculate_memory_usage(model: nn.Module, 
                          batch_size: int = 1,
                          resolution: int = 1008) -> Dict[str, float]:
    """
    Estimate GPU memory usage
    
    Args:
        model: PyTorch model
        batch_size: Training batch size
        resolution: Image resolution
    
    Returns:
        Dictionary with memory estimates in GB
    """
    param_counts = count_parameters(model)
    
    # Parameter memory (4 bytes per float32 parameter)
    param_memory = param_counts['total'] * 4 / (1024 ** 3)
    
    # Gradient memory (only for trainable params)
    gradient_memory = param_counts['trainable'] * 4 / (1024 ** 3)
    
    # Optimizer state memory (Adam: 2x parameters)
    optimizer_memory = param_counts['trainable'] * 4 * 2 / (1024 ** 3)
    
    # Activation memory (rough estimate)
    activation_memory = batch_size * resolution * resolution * 3 * 4 / (1024 ** 3)
    
    total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'parameters': param_memory,
        'gradients': gradient_memory,
        'optimizer_state': optimizer_memory,
        'activations': activation_memory,
        'total_estimated': total_memory
    }


class MetricsTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics with new values
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics"""
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
        return averages
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
