"""Visualization utilities for deepfake detection.

This module provides functions for visualizing DCT features,
adversarial examples, training curves, and feature maps.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def visualize_dct_features(
    images: torch.Tensor,
    dct_extractor: nn.Module,
    save_path: Optional[str] = None,
    num_samples: int = 4
) -> Figure:
    """Visualize original images and their DCT representations.
    
    Args:
        images: Input images tensor [B, 3, H, W].
        dct_extractor: DCTFeatureExtractor module.
        save_path: Optional path to save the figure.
        num_samples: Number of samples to visualize (default: 4).
        
    Returns:
        Matplotlib figure object.
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    # Move to CPU for visualization
    images_cpu = images[:num_samples].cpu()
    
    with torch.no_grad():
        dct_features = dct_extractor(images[:num_samples].to(next(dct_extractor.parameters()).device))
        dct_features_cpu = dct_features.cpu()
    
    for i in range(num_samples):
        # Original image (transpose from CHW to HWC)
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # DCT features (grayscale)
        dct = dct_features_cpu[i, 0].numpy()
        axes[i, 1].imshow(dct, cmap='hot')
        axes[i, 1].set_title(f'DCT Features {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_adversarial_examples(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    perturbation: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    amplify_perturbation: float = 10.0
) -> Figure:
    """Visualize original, perturbation, and adversarial images.
    
    Args:
        original: Original images [B, 3, H, W].
        adversarial: Adversarial images [B, 3, H, W].
        perturbation: Perturbation images [B, 3, H, W].
        save_path: Optional path to save the figure.
        num_samples: Number of samples to visualize (default: 4).
        amplify_perturbation: Factor to amplify perturbation for visibility.
        
    Returns:
        Matplotlib figure object.
    """
    num_samples = min(num_samples, original.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 3)
    
    # Move to CPU
    original_cpu = original[:num_samples].cpu()
    adversarial_cpu = adversarial[:num_samples].cpu()
    perturbation_cpu = perturbation[:num_samples].cpu()
    
    for i in range(num_samples):
        # Original image
        orig_img = original_cpu[i].permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        # Perturbation (amplified and normalized for visibility)
        pert = perturbation_cpu[i].permute(1, 2, 0).numpy()
        pert_amplified = (pert * amplify_perturbation + 0.5)
        pert_amplified = np.clip(pert_amplified, 0, 1)
        axes[i, 1].imshow(pert_amplified)
        axes[i, 1].set_title(f'Perturbation {i+1} (x{amplify_perturbation})')
        axes[i, 1].axis('off')
        
        # Adversarial image
        adv_img = adversarial_cpu[i].permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0, 1)
        axes[i, 2].imshow(adv_img)
        axes[i, 2].set_title(f'Adversarial {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> Figure:
    """Plot training curves for losses and accuracies.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    # Separate losses and accuracies
    loss_metrics = {k: v for k, v in metrics_history.items() if 'loss' in k.lower()}
    acc_metrics = {k: v for k, v in metrics_history.items() if 'acc' in k.lower() or 'f1' in k.lower()}
    
    num_plots = sum([len(loss_metrics) > 0, len(acc_metrics) > 0])
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot losses
    if loss_metrics:
        ax = axes[plot_idx]
        for name, values in loss_metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot accuracies
    if acc_metrics:
        ax = axes[plot_idx]
        for name, values in acc_metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy / F1')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    save_path: Optional[str] = None,
    layer_name: str = 'backbone',
    num_features: int = 16
) -> Figure:
    """Visualize intermediate feature maps from the discriminator.
    
    Args:
        model: The discriminator model.
        image: Input image tensor [1, 3, H, W] or [3, H, W].
        save_path: Optional path to save the figure.
        layer_name: Name of the layer to extract features from.
        num_features: Number of feature maps to visualize.
        
    Returns:
        Matplotlib figure object.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Storage for activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hook on specified layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Warning: Layer '{layer_name}' not found. Using default feature extraction.")
        # Use the model's get_features method if available
        if hasattr(model, 'get_features'):
            with torch.no_grad():
                features = model.get_features(image)
                activations['features'] = features
        else:
            raise ValueError(f"Could not find layer '{layer_name}' and model has no get_features method")
    else:
        handle = target_layer.register_forward_hook(hook_fn('features'))
        with torch.no_grad():
            _ = model(image)
        handle.remove()
    
    # Get features
    features = activations['features']
    if features.dim() == 4:
        features = features[0]  # Remove batch dimension
    
    # Select subset of feature maps to visualize
    num_features = min(num_features, features.size(0))
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        feature_map = features[i].cpu().numpy()
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature {i+1}')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Maps', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: Dict[str, int],
    save_path: Optional[str] = None,
    class_names: List[str] = None
) -> Figure:
    """Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix dict with TP, TN, FP, FN.
        save_path: Optional path to save the figure.
        class_names: Class names (default: ['Fake', 'Real']).
        
    Returns:
        Matplotlib figure object.
    """
    if class_names is None:
        class_names = ['Fake', 'Real']
    
    # Build matrix
    matrix = np.array([
        [cm['TN'], cm['FP']],
        [cm['FN'], cm['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f'Pred {n}' for n in class_names])
    ax.set_yticklabels([f'True {n}' for n in class_names])
    
    # Add values in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), 
                   ha='center', va='center', fontsize=14)
    
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """Plot ROC curve.
    
    Args:
        labels: Ground truth binary labels.
        scores: Prediction scores (probabilities or logits).
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
