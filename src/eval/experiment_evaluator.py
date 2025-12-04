"""
Experiment Evaluator for AgriSAM3

Comprehensive evaluation tool for individual experiments and universal model.
Loads checkpoints, runs inference on test sets, computes metrics, and generates reports.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.universal_finetune import UniversalAgriculturalFineTuner
from src.eval.metrics_universal import AgriculturalMetricsAggregator
from src.training.data_loaders import AgriculturalDatasetLoader
from src.training.agricultural_prompts import AgriculturalPromptGenerator


class ExperimentEvaluator:
    """
    Evaluate trained AgriSAM3 models on test sets
    
    Supports evaluation of:
    - Individual experiment checkpoints
    - Universal model (trained across all experiments)
    - Cross-domain generalization (train on one domain, test on another)
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        """
        Initialize evaluator
        
        Args:
            config_path: Path to experiment configuration YAML
            checkpoint_path: Path to model checkpoint
            output_dir: Directory to save evaluation results
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_name = self.config['experiment']['name']
        self.domain = self.config['experiment']['domain']
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.data_loader = None
        self.metrics_aggregator = None
        
        print(f"Experiment Evaluator initialized")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Domain: {self.domain}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Output: {self.output_dir}")
    
    def load_model(self):
        """Load model from checkpoint"""
        print(f"\nLoading model from: {self.checkpoint_path}")
        
        # Build fine-tuner to access model architecture
        fine_tuner = UniversalAgriculturalFineTuner(self.config, self.experiment_name)
        fine_tuner.setup_model()
        
        # Load checkpoint weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        fine_tuner.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = fine_tuner.model
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    def setup_data_loader(self, split: str = 'test'):
        """
        Setup data loader for evaluation
        
        Args:
            split: Data split to evaluate ('test' or 'val')
        """
        print(f"\nSetting up data loader for {split} split...")
        
        # Initialize prompt generator
        prompt_generator = AgriculturalPromptGenerator(
            concepts=self.config['dataset'].get('concepts', []),
            domain=self.domain
        )
        
        # Create dataset loader
        dataset_loader = AgriculturalDatasetLoader(
            data_config=self.config['dataset'],
            prompt_generator=prompt_generator,
            resolution=self.config['training'].get('resolution', 1008)
        )
        
        dataset = dataset_loader.get_dataset(split)
        
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # Evaluate one at a time for detailed analysis
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Data loader created: {len(self.data_loader)} samples")
    
    def setup_metrics(self):
        """Initialize metrics aggregator"""
        concepts = self.config['dataset'].get('concepts', [])
        
        self.metrics_aggregator = AgriculturalMetricsAggregator(
            concepts=concepts,
            enable_boundary=True,
            enable_multiscale=True
        )
        
        print(f"\nMetrics initialized for concepts: {concepts}")
    
    def evaluate(self, visualize: bool = True, save_predictions: bool = False) -> Dict:
        """
        Run complete evaluation
        
        Args:
            visualize: Whether to generate visualizations
            save_predictions: Whether to save predicted masks
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "="*80)
        print("STARTING EVALUATION")
        print("="*80)
        
        if self.model is None:
            self.load_model()
        if self.data_loader is None:
            self.setup_data_loader()
        if self.metrics_aggregator is None:
            self.setup_metrics()
        
        # Evaluation loop
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
                # Move to device
                images = batch['image_array'].to(self.device)
                masks = batch['masks'].to(self.device)
                text_prompts = batch['text_prompts']
                
                # Forward pass
                predictions = self.model(
                    images=images,
                    text_prompts=text_prompts,
                    enable_segmentation=True
                )
                
                # Extract predictions
                pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                target_masks = masks.cpu().numpy()
                
                # Update metrics
                self.metrics_aggregator.update(
                    predictions={'masks': pred_masks},
                    targets={
                        'masks': target_masks,
                        'instance_masks': [target_masks[0]]
                    }
                )
                
                # Store for visualization
                if visualize and batch_idx < 20:  # Visualize first 20 samples
                    all_predictions.append({
                        'image': batch['image'],
                        'pred_masks': pred_masks[0],
                        'target_masks': target_masks[0],
                        'text_prompts': text_prompts,
                        'metadata': batch['metadata']
                    })
                
                # Save predictions
                if save_predictions:
                    self._save_prediction(batch, pred_masks, batch_idx)
        
        # Compute final metrics
        print("\nComputing metrics...")
        metrics = self.metrics_aggregator.compute()
        
        # Generate visualizations
        if visualize and all_predictions:
            print("\nGenerating visualizations...")
            self._visualize_results(all_predictions)
        
        # Save metrics
        self._save_metrics(metrics)
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _save_prediction(self, batch: Dict, predictions: np.ndarray, idx: int):
        """Save predicted masks to disk"""
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        
        # Save as numpy array
        save_path = pred_dir / f"pred_{idx:04d}.npy"
        np.save(save_path, predictions)
    
    def _visualize_results(self, predictions: List[Dict]):
        """Generate visualization of predictions"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for idx, pred_data in enumerate(predictions[:10]):  # Show first 10
            image = pred_data['image']
            if isinstance(image, list):
                image = image[0]
            
            image_array = np.array(image)
            pred_masks = pred_data['pred_masks']
            target_masks = pred_data['target_masks']
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_array)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(image_array)
            if len(target_masks.shape) == 3:
                # Multiple instances
                combined_mask = target_masks.max(axis=0)
            else:
                combined_mask = target_masks
            axes[1].imshow(combined_mask, alpha=0.5, cmap='Greens')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(image_array)
            if len(pred_masks.shape) == 3:
                combined_pred = pred_masks.max(axis=0)
            else:
                combined_pred = pred_masks
            axes[2].imshow(combined_pred, alpha=0.5, cmap='Reds')
            axes[2].set_title("Prediction")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"sample_{idx:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to JSON file"""
        metrics_file = self.output_dir / "metrics.json"
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        metrics_serializable = convert_types(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")
    
    def _print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Segmentation metrics
        if 'segmentation' in metrics:
            seg = metrics['segmentation']
            print("\nSegmentation Metrics:")
            print(f"  IoU:       {seg['iou']:.4f}")
            print(f"  Dice:      {seg['dice']:.4f}")
            print(f"  Precision: {seg['precision']:.4f}")
            print(f"  Recall:    {seg['recall']:.4f}")
            print(f"  F1:        {seg['f1']:.4f}")
            print(f"  Accuracy:  {seg['accuracy']:.4f}")
        
        # Concept metrics
        if 'concepts' in metrics:
            print("\nConcept Recognition:")
            for concept, concept_metrics in metrics['concepts'].items():
                print(f"  {concept}:")
                print(f"    Accuracy: {concept_metrics['accuracy']:.4f}")
        
        # Boundary metrics
        if 'boundary' in metrics:
            print("\nBoundary Detection:")
            print(f"  Boundary F1: {metrics['boundary']['boundary_f1']:.4f}")
        
        # Multi-scale metrics
        if 'multiscale' in metrics:
            print("\nMulti-Scale Performance:")
            for scale, scale_metrics in metrics['multiscale'].items():
                print(f"  {scale.capitalize()} objects:")
                print(f"    IoU: {scale_metrics['iou']:.4f}")
        
        print("="*80)
    
    def compare_with_baseline(self, baseline_metrics_path: str) -> Dict:
        """
        Compare current results with baseline
        
        Args:
            baseline_metrics_path: Path to baseline metrics JSON
        
        Returns:
            Dictionary with comparison results
        """
        with open(baseline_metrics_path, 'r') as f:
            baseline = json.load(f)
        
        current = self.metrics_aggregator.compute()
        
        comparison = {}
        
        # Compare segmentation metrics
        if 'segmentation' in baseline and 'segmentation' in current:
            comparison['segmentation'] = {}
            for metric in ['iou', 'dice', 'f1']:
                baseline_val = baseline['segmentation'][metric]
                current_val = current['segmentation'][metric]
                improvement = current_val - baseline_val
                comparison['segmentation'][metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'improvement': improvement,
                    'improvement_pct': (improvement / baseline_val) * 100
                }
        
        return comparison


class UniversalModelEvaluator:
    """
    Evaluate universal model across all experiments
    
    Tests generalization capability by evaluating single model on all domains.
    """
    
    def __init__(self, universal_checkpoint: str, experiment_configs: List[str],
                 output_dir: str):
        """
        Initialize universal evaluator
        
        Args:
            universal_checkpoint: Path to universal model checkpoint
            experiment_configs: List of paths to experiment configs
            output_dir: Directory for evaluation results
        """
        self.checkpoint_path = Path(universal_checkpoint)
        self.experiment_configs = [Path(p) for p in experiment_configs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Universal Model Evaluator")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Experiments: {len(self.experiment_configs)}")
    
    def evaluate_all_domains(self) -> Dict:
        """
        Evaluate universal model on all experiment domains
        
        Returns:
            Dictionary with per-domain and aggregate metrics
        """
        results = {}
        
        for config_path in self.experiment_configs:
            exp_name = config_path.stem  # e.g., 'exp01_fruit_ripeness'
            
            print(f"\n{'='*80}")
            print(f"Evaluating on: {exp_name}")
            print(f"{'='*80}")
            
            # Create evaluator for this experiment
            exp_output = self.output_dir / exp_name
            evaluator = ExperimentEvaluator(
                config_path=str(config_path),
                checkpoint_path=str(self.checkpoint_path),
                output_dir=str(exp_output)
            )
            
            # Run evaluation
            metrics = evaluator.evaluate(visualize=True)
            results[exp_name] = metrics
        
        # Aggregate results
        aggregate = self._aggregate_results(results)
        
        # Save aggregate
        with open(self.output_dir / "universal_results.json", 'w') as f:
            json.dump({
                'per_domain': results,
                'aggregate': aggregate
            }, f, indent=2)
        
        # Print summary
        self._print_universal_summary(results, aggregate)
        
        return {'per_domain': results, 'aggregate': aggregate}
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate metrics across all domains"""
        aggregate = {
            'segmentation': {},
            'num_domains': len(results)
        }
        
        # Average segmentation metrics
        seg_metrics = ['iou', 'dice', 'precision', 'recall', 'f1']
        for metric in seg_metrics:
            values = [
                results[exp]['segmentation'][metric]
                for exp in results
                if 'segmentation' in results[exp]
            ]
            if values:
                aggregate['segmentation'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregate
    
    def _print_universal_summary(self, results: Dict, aggregate: Dict):
        """Print universal model summary"""
        print("\n" + "="*80)
        print("UNIVERSAL MODEL SUMMARY")
        print("="*80)
        
        print(f"\nEvaluated on {aggregate['num_domains']} domains")
        
        print("\nAverage Performance:")
        for metric, stats in aggregate['segmentation'].items():
            print(f"  {metric.upper()}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("\nPer-Domain Performance:")
        for exp_name, exp_results in results.items():
            if 'segmentation' in exp_results:
                iou = exp_results['segmentation']['iou']
                print(f"  {exp_name}: IoU = {iou:.4f}")
        
        print("="*80)


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description="Evaluate AgriSAM3 Experiments")
    parser.add_argument('--mode', type=str, choices=['single', 'universal'], required=True,
                       help='Evaluation mode')
    parser.add_argument('--config', type=str, help='Config file for single experiment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_configs', nargs='+', help='Config files for universal eval')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single experiment evaluation
        evaluator = ExperimentEvaluator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output
        )
        metrics = evaluator.evaluate(visualize=args.visualize)
    
    elif args.mode == 'universal':
        # Universal model evaluation
        evaluator = UniversalModelEvaluator(
            universal_checkpoint=args.checkpoint,
            experiment_configs=args.experiment_configs,
            output_dir=args.output
        )
        results = evaluator.evaluate_all_domains()


if __name__ == "__main__":
    main()
