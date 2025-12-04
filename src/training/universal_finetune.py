"""
Universal Fine-tuning Framework for AgriSAM3

This module provides a flexible, reusable training infrastructure that works with
any agricultural dataset. It automatically adapts to different:
- Dataset formats (COCO, Pascal VOC, custom JSON)
- Agricultural concepts (crops, weeds, diseases, pests, ripeness, etc.)
- Annotation schemas (instance segmentation, semantic segmentation)
- Text prompt templates (auto-generated from agricultural attributes)

Based on official SAM3 training guidelines and optimized for agricultural applications.

Usage:
    python src/training/universal_finetune.py \\
        --experiment exp01_fruit_ripeness \\
        --config configs/exp01_fruit_ripeness.yml \\
        --resume_from pretrained

Key Features:
- Modular data loading for multiple formats
- Automatic agricultural text prompt generation
- Progressive training across experiments
- Memory-efficient training strategies
- Comprehensive logging and checkpointing
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import sam3
    from sam3.build_sam3 import build_sam3_model
    SAM3_AVAILABLE = True
except ImportError:
    warnings.warn("SAM3 not installed. Install: pip install git+https://github.com/facebookresearch/sam3.git")
    SAM3_AVAILABLE = False

from src.training.data_loaders import AgriculturalDatasetLoader
from src.training.agricultural_prompts import AgriculturalPromptGenerator
from src.training.training_utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    compute_metrics
)


class UniversalAgriculturalFineTuner:
    """
    Universal fine-tuning framework for SAM3 on agricultural datasets
    
    Designed to work with any agricultural segmentation task through:
    - Flexible data loading (supports COCO, VOC, custom formats)
    - Automatic prompt generation from agricultural concepts
    - Modular architecture for easy experiment addition
    - Progressive training support (initialize from previous experiments)
    """
    
    def __init__(self, config: Dict, experiment_name: str):
        """
        Initialize universal fine-tuner
        
        Args:
            config: Complete configuration dictionary from YAML
            experiment_name: Unique experiment identifier (e.g., "exp01_fruit_ripeness")
        """
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration sections
        self.exp_config = config['experiment']
        self.data_config = config['dataset']
        self.train_config = config['training']
        self.eval_config = config.get('evaluation', {})
        
        # Setup output directory
        self.output_dir = Path(config['output']['save_dir']) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_epochs = self.train_config['epochs']
        self.batch_size = self.train_config['batch_size']
        self.learning_rate = self.train_config['learning_rate']
        self.weight_decay = self.train_config.get('weight_decay', 0.05)
        self.gradient_clip = self.train_config.get('gradient_clip', 1.0)
        self.mixed_precision = self.train_config.get('mixed_precision', True)
        self.gradient_accumulation = self.train_config.get('gradient_accumulation', 4)
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Prompt generator for agricultural concepts
        self.prompt_generator = AgriculturalPromptGenerator(
            concepts=self.data_config.get('concepts', []),
            domain=self.exp_config.get('domain', 'general_agriculture')
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'segmentation_loss': [],
            'grounding_loss': [],
            'learning_rate': []
        }
        
        # Setup logging
        setup_logging(self.output_dir, experiment_name)
        
        self.log(f"Initialized Universal Agricultural Fine-Tuner")
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Domain: {self.exp_config.get('domain')}")
        self.log(f"Dataset: {self.data_config['name']}")
        self.log(f"Output: {self.output_dir}")
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    def setup_model(self):
        """
        Load SAM3 model and configure for agricultural fine-tuning
        
        Follows SAM3 official guidelines:
        - Train detector + shared backbone
        - Freeze tracker modules
        - Optional: freeze vision/text encoders for memory efficiency
        """
        if not SAM3_AVAILABLE:
            raise RuntimeError("SAM3 not available")
        
        self.log(f"Loading SAM3 model...")
        
        # Get checkpoint path
        resume_from = self.train_config.get('resume_from', 'pretrained')
        
        if resume_from == 'pretrained':
            # Load official pretrained SAM3
            checkpoint_path = self.config.get('model', {}).get('checkpoint')
            self.log(f"Loading pretrained SAM3 from: {checkpoint_path}")
        else:
            # Resume from previous experiment checkpoint
            checkpoint_path = resume_from
            self.log(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Build model
        self.model = build_sam3_model(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device)
        
        # Configure trainable parameters
        freeze_vision = self.train_config.get('freeze_vision_encoder', False)
        freeze_text = self.train_config.get('freeze_text_encoder', False)
        freeze_tracker = True  # Always freeze tracker per SAM3 design
        
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Freeze tracker (per SAM3 guidelines)
            if 'tracker' in name.lower() or 'memory' in name.lower():
                param.requires_grad = False
            # Optionally freeze vision encoder
            elif freeze_vision and 'vision_encoder' in name:
                param.requires_grad = False
            # Optionally freeze text encoder
            elif freeze_text and 'text_encoder' in name:
                param.requires_grad = False
            # Train detector and backbone
            else:
                param.requires_grad = True
                trainable_params += param.numel()
        
        self.log(f"Model Parameters:")
        self.log(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
        self.log(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M, {100*trainable_params/total_params:.1f}%)")
        self.log(f"  Frozen: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
        
        self.log(f"Module Status:")
        self.log(f"  Vision Encoder: {'FROZEN' if freeze_vision else 'TRAINABLE'}")
        self.log(f"  Text Encoder: {'FROZEN' if freeze_text else 'TRAINABLE'}")
        self.log(f"  Detector: TRAINABLE")
        self.log(f"  Tracker: FROZEN (per SAM3 design)")
        
        # Set frozen modules to eval mode
        self._set_frozen_modules_eval()
    
    def _set_frozen_modules_eval(self):
        """Set frozen modules to evaluation mode"""
        freeze_vision = self.train_config.get('freeze_vision_encoder', False)
        freeze_text = self.train_config.get('freeze_text_encoder', False)
        
        if freeze_vision:
            for name, module in self.model.named_modules():
                if 'vision_encoder' in name:
                    module.eval()
        
        if freeze_text:
            for name, module in self.model.named_modules():
                if 'text_encoder' in name:
                    module.eval()
        
        # Always freeze tracker
        for name, module in self.model.named_modules():
            if 'tracker' in name.lower() or 'memory' in name.lower():
                module.eval()
    
    def setup_optimizer(self):
        """Configure optimizer and learning rate scheduler"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        warmup_epochs = self.train_config.get('warmup_epochs', 2)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (self.num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.log(f"Optimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        self.log(f"Scheduler: Cosine annealing with {warmup_epochs} warmup epochs")
        self.log(f"Gradient accumulation: {self.gradient_accumulation} steps")
    
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for training, validation, and testing
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.log(f"Setting up data loaders...")
        
        # Universal dataset loader handles multiple formats
        dataset_loader = AgriculturalDatasetLoader(
            data_config=self.data_config,
            prompt_generator=self.prompt_generator,
            resolution=self.train_config.get('resolution', 1008)
        )
        
        train_dataset = dataset_loader.get_dataset('train')
        val_dataset = dataset_loader.get_dataset('val')
        test_dataset = dataset_loader.get_dataset('test')
        
        self.log(f"Dataset Statistics:")
        self.log(f"  Train: {len(train_dataset)} images")
        self.log(f"  Val: {len(val_dataset)} images")
        self.log(f"  Test: {len(test_dataset)} images")
        
        # Create data loaders
        num_workers = self.train_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def compute_loss(self, predictions: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for agricultural segmentation
        
        Args:
            predictions: Model predictions
            batch: Ground truth batch
        
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Extract predictions and targets
        pred_masks = predictions.get('masks')
        target_masks = batch['masks'].to(self.device)
        
        # Segmentation loss (focal + dice)
        seg_loss = self._segmentation_loss(pred_masks, target_masks)
        losses['segmentation'] = seg_loss * self.train_config['loss_weights'].get('segmentation', 1.0)
        
        # Vision-language grounding loss
        if 'text_embeddings' in predictions and 'visual_embeddings' in predictions:
            grounding_loss = self._grounding_loss(
                predictions['visual_embeddings'],
                predictions['text_embeddings'],
                target_masks
            )
            losses['grounding'] = grounding_loss * self.train_config['loss_weights'].get('grounding', 0.5)
        
        # Domain-specific losses (e.g., ripeness classification, disease detection)
        if 'domain_logits' in predictions and 'domain_labels' in batch:
            domain_loss = nn.functional.cross_entropy(
                predictions['domain_logits'],
                batch['domain_labels'].to(self.device)
            )
            losses['domain'] = domain_loss * self.train_config['loss_weights'].get('domain', 0.3)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _segmentation_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Combined segmentation loss: focal + dice"""
        focal = self._focal_loss(pred_masks, target_masks)
        dice = self._dice_loss(pred_masks, target_masks)
        return focal + dice
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for class imbalance"""
        probs = torch.sigmoid(predictions)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** gamma
        bce = nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        focal = focal_weight * bce
        focal = torch.where(targets == 1, alpha * focal, (1 - alpha) * focal)
        return focal.mean()
    
    def _dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for overlap optimization"""
        probs = torch.sigmoid(predictions)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        return 1.0 - dice
    
    def _grounding_loss(self, visual_embeddings: torch.Tensor,
                       text_embeddings: torch.Tensor,
                       masks: torch.Tensor) -> torch.Tensor:
        """Vision-language grounding loss"""
        visual_norm = nn.functional.normalize(visual_embeddings, dim=-1)
        text_norm = nn.functional.normalize(text_embeddings, dim=-1)
        similarity = torch.matmul(visual_norm, text_norm.transpose(-2, -1))
        target = torch.eye(similarity.size(0), device=self.device)
        loss = nn.functional.cross_entropy(similarity / 0.07, target.argmax(dim=1))
        return loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        self._set_frozen_modules_eval()
        
        total_loss = 0.0
        loss_components = {'segmentation': 0.0, 'grounding': 0.0, 'domain': 0.0}
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image_array'].to(self.device)
            masks = batch['masks'].to(self.device)
            text_prompts = batch['text_prompts']
            
            # Forward pass
            with autocast(enabled=self.mixed_precision):
                predictions = self.model(
                    images=images,
                    text_prompts=text_prompts,
                    enable_segmentation=True
                )
                
                losses = self.compute_loss(predictions, batch)
                loss = losses['total'] / self.gradient_accumulation
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'avg': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image_array'].to(self.device)
                masks = batch['masks'].to(self.device)
                text_prompts = batch['text_prompts']
                
                predictions = self.model(
                    images=images,
                    text_prompts=text_prompts,
                    enable_segmentation=True
                )
                
                losses = self.compute_loss(predictions, batch)
                total_loss += losses['total'].item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        self.log(f"Starting training for {self.num_epochs} epochs")
        self.log("=" * 80)
        
        for epoch in range(self.current_epoch + 1, self.num_epochs + 1):
            # Train
            train_loss, loss_components = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['segmentation_loss'].append(loss_components['segmentation'])
            self.training_history['grounding_loss'].append(loss_components['grounding'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Print summary
            self.log(f"Epoch {epoch}/{self.num_epochs} Summary:")
            self.log(f"  Train Loss: {train_loss:.4f}")
            self.log(f"  Val Loss: {val_loss:.4f}")
            self.log(f"  Seg Loss: {loss_components['segmentation']:.4f}")
            self.log(f"  Ground Loss: {loss_components['grounding']:.4f}")
            self.log(f"  LR: {current_lr:.6f}")
            if is_best:
                self.log(f"  ✓ New best model!")
            self.log("=" * 80)
            
            # Save checkpoint
            if epoch % self.train_config.get('save_frequency', 5) == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_val_loss,
                    self.training_history,
                    self.output_dir,
                    is_best
                )
            
            # Save history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        self.log("Training completed!")
        self.log(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Universal AgriSAM3 Fine-tuning")
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., exp01_fruit_ripeness)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config YAML')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override resume_from if provided
    if args.resume_from:
        config['training']['resume_from'] = args.resume_from
    
    print("=" * 80)
    print("AgriSAM3 Universal Fine-tuning")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create fine-tuner
    fine_tuner = UniversalAgriculturalFineTuner(config, args.experiment)
    
    # Setup model and optimizer
    fine_tuner.setup_model()
    fine_tuner.setup_optimizer()
    
    # Setup data
    train_loader, val_loader, test_loader = fine_tuner.setup_dataloaders()
    
    # Train
    fine_tuner.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
