"""Metrics computation utilities for deepfake detection.

This module provides functions for computing classification metrics
and a tracker class for aggregating metrics across batches.
"""

from typing import Dict, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary classification accuracy.
    
    Args:
        predictions: Binary predictions tensor.
        labels: Ground truth labels tensor.
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    predictions = predictions.float()
    labels = labels.float()
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0


def compute_confusion_matrix(
    predictions: torch.Tensor, 
    labels: torch.Tensor
) -> Dict[str, int]:
    """Compute confusion matrix components.
    
    Args:
        predictions: Binary predictions tensor.
        labels: Ground truth labels tensor.
        
    Returns:
        Dictionary with TP, TN, FP, FN counts.
    """
    predictions = predictions.float()
    labels = labels.float()
    
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    
    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}


def compute_precision_recall_f1(
    predictions: torch.Tensor, 
    labels: torch.Tensor
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score.
    
    Args:
        predictions: Binary predictions tensor.
        labels: Ground truth labels tensor.
        
    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    cm = compute_confusion_matrix(predictions, labels)
    tp, fp, fn = cm['TP'], cm['FP'], cm['FN']
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_roc_auc(
    predictions: torch.Tensor, 
    labels: torch.Tensor,
    scores: Optional[torch.Tensor] = None
) -> float:
    """Compute ROC-AUC score.
    
    Args:
        predictions: Binary predictions tensor (unused if scores provided).
        labels: Ground truth labels tensor.
        scores: Optional continuous scores (logits or probabilities).
                If not provided, uses predictions.
        
    Returns:
        ROC-AUC score as a float.
    """
    labels_np = labels.cpu().numpy()
    
    if scores is not None:
        scores_np = scores.cpu().numpy()
    else:
        scores_np = predictions.cpu().numpy()
    
    # Handle edge case where only one class is present
    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2:
        return 0.5  # Return 0.5 (random) if only one class
    
    try:
        return roc_auc_score(labels_np, scores_np)
    except ValueError:
        return 0.5


class MetricsTracker:
    """Track and aggregate metrics across training batches.
    
    Stores running metrics and provides methods to compute averages
    and pretty-print results.
    """
    
    def __init__(self) -> None:
        self._metrics: Dict[str, list] = {}
        self._counts: Dict[str, int] = {}
    
    def update(self, metrics_dict: Dict[str, float]) -> None:
        """Update tracker with new metrics.
        
        Args:
            metrics_dict: Dictionary of metric names to values.
        """
        for key, value in metrics_dict.items():
            if key not in self._metrics:
                self._metrics[key] = []
                self._counts[key] = 0
            self._metrics[key].append(value)
            self._counts[key] += 1
    
    def get_average(self) -> Dict[str, float]:
        """Get average values for all tracked metrics.
        
        Returns:
            Dictionary of metric names to average values.
        """
        averages = {}
        for key, values in self._metrics.items():
            if len(values) > 0:
                averages[key] = sum(values) / len(values)
            else:
                averages[key] = 0.0
        return averages
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values for all tracked metrics.
        
        Returns:
            Dictionary of metric names to latest values.
        """
        latest = {}
        for key, values in self._metrics.items():
            if len(values) > 0:
                latest[key] = values[-1]
            else:
                latest[key] = 0.0
        return latest
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._metrics = {}
        self._counts = {}
    
    def pretty_print(self, prefix: str = "") -> str:
        """Generate pretty-printed string of average metrics.
        
        Args:
            prefix: Optional prefix for the output string.
            
        Returns:
            Formatted string of metrics.
        """
        averages = self.get_average()
        lines = [prefix] if prefix else []
        
        for key, value in sorted(averages.items()):
            if 'loss' in key.lower():
                lines.append(f"  {key}: {value:.4f}")
            elif 'acc' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower() or 'f1' in key.lower():
                lines.append(f"  {key}: {value:.2%}")
            else:
                lines.append(f"  {key}: {value:.4f}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.pretty_print()
    
    def __repr__(self) -> str:
        return f"MetricsTracker(metrics={list(self._metrics.keys())})"
