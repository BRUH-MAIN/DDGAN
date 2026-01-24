"""Evaluation script for Deepfake Detection GAN.

This script evaluates a trained discriminator model on test/validation data,
computing metrics and generating visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --dataset_name "your-dataset/name"
    python evaluate.py --help
"""

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from config import get_default_config
from src.data.dataset import load_deepfake_dataset
from src.data.transforms import collate_fn, get_gpu_transform
from src.models.discriminator import DCTDiscriminator
from src.utils.metrics import (
    compute_accuracy, 
    compute_precision_recall_f1, 
    compute_confusion_matrix,
    compute_roc_auc
)
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    visualize_dct_features
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory for evaluation outputs")
    parser.add_argument("--no_viz", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for datasets")
    
    return parser.parse_args()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    gpu_transform: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Dict:
    """Evaluate model on a dataset.
    
    Args:
        model: The discriminator model.
        dataloader: DataLoader for evaluation data.
        gpu_transform: GPU transforms to apply.
        device: Torch device.
        use_amp: Whether to use mixed precision.
        
    Returns:
        Dictionary containing all evaluation metrics and data.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Apply GPU transforms
            images = gpu_transform(images)
            
            # Forward pass
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            
            # Convert to predictions and scores
            scores = torch.sigmoid(logits).squeeze()
            predictions = (logits > 0).float().squeeze()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
            
            # Store some images for visualization (first 100)
            if len(all_images) < 100:
                all_images.append(images.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    all_images = torch.cat(all_images)[:100]  # Limit to 100 images
    
    # Compute metrics
    accuracy = compute_accuracy(all_predictions, all_labels)
    prf1 = compute_precision_recall_f1(all_predictions, all_labels)
    cm = compute_confusion_matrix(all_predictions, all_labels)
    roc_auc = compute_roc_auc(all_predictions, all_labels, all_scores)
    
    # Find misclassified samples
    misclassified_mask = all_predictions != all_labels
    misclassified_indices = torch.where(misclassified_mask)[0]
    
    # Find correct samples by class
    correct_real_mask = (all_predictions == 1) & (all_labels == 1)
    correct_fake_mask = (all_predictions == 0) & (all_labels == 0)
    
    return {
        'accuracy': accuracy,
        'precision': prf1['precision'],
        'recall': prf1['recall'],
        'f1': prf1['f1'],
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'total_samples': len(all_labels),
        'predictions': all_predictions,
        'labels': all_labels,
        'scores': all_scores,
        'images': all_images,
        'misclassified_indices': misclassified_indices,
        'num_misclassified': len(misclassified_indices),
        'num_correct_real': correct_real_mask.sum().item(),
        'num_correct_fake': correct_fake_mask.sum().item(),
    }


def generate_classification_report(results: Dict) -> str:
    """Generate a text classification report.
    
    Args:
        results: Dictionary of evaluation results.
        
    Returns:
        Formatted report string.
    """
    cm = results['confusion_matrix']
    
    report = []
    report.append("=" * 60)
    report.append("DEEPFAKE DETECTION - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("OVERALL METRICS")
    report.append("-" * 40)
    report.append(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    report.append(f"  Precision: {results['precision']:.4f}")
    report.append(f"  Recall:    {results['recall']:.4f}")
    report.append(f"  F1 Score:  {results['f1']:.4f}")
    report.append(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    report.append("")
    report.append("CONFUSION MATRIX")
    report.append("-" * 40)
    report.append(f"  True Positives (TP):  {cm['TP']}")
    report.append(f"  True Negatives (TN):  {cm['TN']}")
    report.append(f"  False Positives (FP): {cm['FP']}")
    report.append(f"  False Negatives (FN): {cm['FN']}")
    report.append("")
    report.append("SAMPLE STATISTICS")
    report.append("-" * 40)
    report.append(f"  Total Samples:        {results['total_samples']}")
    report.append(f"  Misclassified:        {results['num_misclassified']}")
    report.append(f"  Correct Real (TP):    {results['num_correct_real']}")
    report.append(f"  Correct Fake (TN):    {results['num_correct_fake']}")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def load_discriminator_from_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load discriminator from either Lightning or legacy checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Torch device to load model on.
        
    Returns:
        Loaded discriminator model.
    """
    from src.training.trainer import DeepfakeGANModule
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if it's a Lightning checkpoint
    if 'state_dict' in checkpoint and any(k.startswith('discriminator.') for k in checkpoint['state_dict'].keys()):
        # Lightning checkpoint - load via module
        print("Detected Lightning checkpoint format")
        module = DeepfakeGANModule.load_from_checkpoint(checkpoint_path, map_location=device)
        model = module.discriminator
    elif 'discriminator_state_dict' in checkpoint:
        # Legacy checkpoint format
        print("Detected legacy checkpoint format")
        model = DCTDiscriminator(pretrained=False)
        model.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
        # Try direct state dict
        print("Attempting direct state dict load")
        model = DCTDiscriminator(pretrained=False)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    config = get_default_config()
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Initialize model - supports both Lightning and legacy checkpoints
    print("Initializing model...")
    model = load_discriminator_from_checkpoint(args.checkpoint, device)
    
    # Create GPU transform
    gpu_transform = get_gpu_transform().to(device)
    
    # Load dataset
    dataset_name = args.dataset_name or config.dataset.dataset_name
    print(f"Loading dataset: {dataset_name}")
    dataset = load_deepfake_dataset(dataset_name, cache_dir=args.cache_dir)
    
    # Determine which split to use
    split = args.split
    if split == 'test' and 'test' not in dataset:
        print(f"Warning: 'test' split not found, using 'val' instead")
        split = 'val'
    
    # Set up dataset - transforms applied on-the-fly in collate_fn
    print("Setting up dataset (transforms applied on-the-fly)...")
    eval_dataset = dataset[split]
    eval_dataset.set_format(columns=['image', 'label'])
    
    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(eval_dataset)} samples...")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        gpu_transform=gpu_transform,
        device=device,
        use_amp=config.training.use_amp
    )
    
    # Generate and print report
    report = generate_classification_report(results)
    print("\n" + report)
    
    # Save report to file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    
    # Save metrics to JSON
    metrics_dict = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'roc_auc': results['roc_auc'],
        'confusion_matrix': results['confusion_matrix'],
        'total_samples': results['total_samples'],
        'num_misclassified': results['num_misclassified'],
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'split': split
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Generate visualizations
    if not args.no_viz:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        print(f"  Confusion matrix: {cm_path}")
        
        # ROC curve
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        plot_roc_curve(
            results['labels'].numpy(),
            results['scores'].numpy(),
            roc_path
        )
        print(f"  ROC curve: {roc_path}")
        
        # DCT features visualization
        if len(results['images']) > 0:
            dct_path = os.path.join(args.output_dir, "dct_features.png")
            dct_extractor = model.dct_extractor
            visualize_dct_features(
                results['images'][:4],
                dct_extractor,
                dct_path
            )
            print(f"  DCT features: {dct_path}")
        
        # Visualize some misclassified examples
        if results['num_misclassified'] > 0:
            import matplotlib.pyplot as plt
            
            # Get indices of misclassified samples that are in our stored images
            mis_indices = results['misclassified_indices']
            mis_indices = mis_indices[mis_indices < len(results['images'])]
            
            if len(mis_indices) > 0:
                num_show = min(8, len(mis_indices))
                fig, axes = plt.subplots(2, num_show // 2, figsize=(12, 8))
                axes = axes.flatten()
                
                for i, idx in enumerate(mis_indices[:num_show]):
                    img = results['images'][idx].permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[i].imshow(img)
                    
                    true_label = "Real" if results['labels'][idx] == 1 else "Fake"
                    pred_label = "Real" if results['predictions'][idx] == 1 else "Fake"
                    score = results['scores'][idx].item()
                    
                    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({score:.2f})")
                    axes[i].axis('off')
                
                plt.suptitle("Misclassified Examples", fontsize=14)
                plt.tight_layout()
                
                mis_path = os.path.join(args.output_dir, "misclassified_examples.png")
                plt.savefig(mis_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Misclassified examples: {mis_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
