"""Advanced loss functions for deepfake detection with imbalanced data.

This module provides specialized loss functions designed to handle:
1. Class imbalance (real vs fake samples)
2. Adversarial training robustness
3. Angular margin losses for better feature discrimination

Key implementations:
- FocalLoss: Down-weights easy examples, focuses on hard ones
- WeightedBCELoss: Class-weighted binary cross entropy
- AAMLoss: Additive Angular Margin Loss for deepfake detection
- CombinedImbalanceLoss: Combines multiple strategies
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focal Loss down-weights well-classified examples and focuses on hard,
    misclassified examples. Originally proposed in "Focal Loss for Dense 
    Object Detection" (Lin et al., 2017).
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the correct class
    - α_t is the class balancing weight
    - γ (gamma) is the focusing parameter (γ ≥ 0)
    
    Args:
        alpha: Class balancing weight. Can be a scalar or tensor of shape [2].
               Higher alpha for minority class increases its importance.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
               Recommended: gamma=2.0
        reduction: Reduction method ('mean', 'sum', 'none')
    
    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = model(images)  # [B, 1]
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Raw predictions (before sigmoid) of shape [B, 1] or [B]
            targets: Ground truth labels of shape [B, 1] or [B], values in {0, 1}
            
        Returns:
            Focal loss value.
        """
        # Ensure proper shapes
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t (class weight)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy part
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Combine
        loss = alpha_t * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for class imbalance.
    
    Applies different weights to positive (real) and negative (fake) classes
    to compensate for class imbalance.
    
    Args:
        pos_weight: Weight for positive class (real samples).
                   Set higher if real samples are minority.
        reduction: Reduction method ('mean', 'sum', 'none')
    
    Example:
        >>> # If fake:real ratio is 5:1, set pos_weight=5.0
        >>> loss_fn = WeightedBCELoss(pos_weight=5.0)
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.register_buffer(
            'pos_weight', 
            torch.tensor([pos_weight])
        )
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted BCE loss.
        
        Args:
            logits: Raw predictions of shape [B, 1] or [B]
            targets: Ground truth labels of shape [B, 1] or [B]
            
        Returns:
            Weighted BCE loss value.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction=self.reduction
        )


class AAMLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace-style) for Deepfake Detection.
    
    This loss combines the discriminative power of angular margin losses
    (like ArcFace/CosFace) with binary classification for deepfake detection.
    
    The key idea is to embed features on a hypersphere and add an angular
    margin penalty to increase inter-class variance and reduce intra-class
    variance, making the model more robust to adversarial perturbations.
    
    For binary classification:
    L = -log(exp(s * cos(θ_y + m)) / (exp(s * cos(θ_y + m)) + exp(s * cos(θ_neg))))
    
    where:
    - θ_y is the angle between feature and positive class center
    - m is the angular margin
    - s is the scaling factor
    
    Args:
        in_features: Dimension of input features
        scale: Scaling factor (s). Higher = sharper decision boundary.
               Typical values: 30-64
        margin: Angular margin (m) in radians. Typical values: 0.3-0.5
        easy_margin: If True, uses easier margin formula
        class_weights: Optional weights for real/fake classes [fake_w, real_w]
    
    References:
        - ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        - Adapting ArcFace for deepfake detection improves generalization
    """
    
    def __init__(
        self,
        in_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class centers (weight vectors)
        # Shape: [2, in_features] for binary classification
        self.weight = nn.Parameter(torch.FloatTensor(2, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos/sin values for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Class weights for imbalance
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(2))
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute AAML loss.
        
        Args:
            features: Normalized feature embeddings of shape [B, in_features]
            labels: Ground truth labels of shape [B], values in {0, 1}
            
        Returns:
            Tuple of (loss, logits) where logits can be used for inference.
        """
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity (cos θ)
        # Shape: [B, 2]
        cosine = F.linear(features, weight)
        
        # Compute sin θ from cos θ
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        # Compute cos(θ + m) using angle addition formula
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding
        labels_long = labels.long()
        one_hot = F.one_hot(labels_long, num_classes=2).float()
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        # Compute cross-entropy loss with class weights
        loss = F.cross_entropy(
            output, 
            labels_long,
            weight=self.class_weights.to(features.device)
        )
        
        # Return both loss and logits (for inference)
        # For binary classification, return difference as logit
        logits = output[:, 1] - output[:, 0]  # Positive = real
        
        return loss, logits


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for heavily imbalanced binary classification.
    
    Extends Focal Loss with asymmetric focusing for positive and negative
    samples. Particularly useful when minority class needs stronger emphasis.
    
    Args:
        gamma_pos: Focusing parameter for positive (minority) class
        gamma_neg: Focusing parameter for negative (majority) class
        alpha: Class balancing weight for positive class
        clip: Probability clipping for numerical stability
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        alpha: float = 0.25,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.clip = clip
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute asymmetric focal loss."""
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Clip probabilities for numerical stability
        probs_pos = probs_pos.clamp(min=self.clip)
        probs_neg = probs_neg.clamp(min=self.clip)
        
        # Asymmetric focusing
        pos_loss = targets * self.alpha * ((1 - probs_pos) ** self.gamma_pos) * torch.log(probs_pos)
        neg_loss = (1 - targets) * (1 - self.alpha) * (probs_neg ** self.gamma_neg) * torch.log(probs_neg)
        
        loss = -pos_loss - neg_loss
        return loss.mean()


class CombinedImbalanceLoss(nn.Module):
    """Combined loss function for handling imbalanced deepfake detection.
    
    This loss combines multiple strategies:
    1. Focal Loss: Focus on hard examples
    2. AAML: Angular margin for better discrimination
    3. Class weighting: Compensate for imbalance
    
    The combination provides robust training on imbalanced datasets.
    
    Args:
        in_features: Feature dimension for AAML (set to 0 to disable)
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        aaml_scale: Scale for AAML (if enabled)
        aaml_margin: Margin for AAML (if enabled)
        aaml_weight: Weight for AAML loss component
        class_weights: Optional class weights [fake_weight, real_weight]
    """
    
    def __init__(
        self,
        in_features: int = 0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aaml_scale: float = 30.0,
        aaml_margin: float = 0.5,
        aaml_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.aaml_weight = aaml_weight
        self.use_aaml = in_features > 0
        
        if self.use_aaml:
            self.aaml_loss = AAMLoss(
                in_features=in_features,
                scale=aaml_scale,
                margin=aaml_margin,
                class_weights=class_weights,
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            logits: Raw predictions of shape [B, 1] or [B]
            targets: Ground truth labels of shape [B]
            features: Optional feature embeddings for AAML [B, in_features]
            
        Returns:
            Dictionary with 'total', 'focal', and optionally 'aaml' losses.
        """
        focal = self.focal_loss(logits, targets)
        
        result = {
            'focal': focal,
            'total': focal,
        }
        
        if self.use_aaml and features is not None:
            aaml, _ = self.aaml_loss(features, targets)
            result['aaml'] = aaml
            result['total'] = (1 - self.aaml_weight) * focal + self.aaml_weight * aaml
        
        return result


def compute_class_weights(
    n_fake: int, 
    n_real: int, 
    method: str = 'inverse'
) -> torch.Tensor:
    """Compute class weights from sample counts.
    
    Args:
        n_fake: Number of fake samples
        n_real: Number of real samples
        method: Weighting method:
            - 'inverse': Inverse frequency weighting
            - 'sqrt_inverse': Square root of inverse frequency
            - 'effective': Effective number weighting (for long-tail)
            
    Returns:
        Tensor of shape [2] with [fake_weight, real_weight]
    """
    total = n_fake + n_real
    
    if method == 'inverse':
        # Standard inverse frequency
        w_fake = total / (2 * n_fake)
        w_real = total / (2 * n_real)
    elif method == 'sqrt_inverse':
        # Smoothed inverse frequency
        w_fake = math.sqrt(total / (2 * n_fake))
        w_real = math.sqrt(total / (2 * n_real))
    elif method == 'effective':
        # Effective number weighting (better for extreme imbalance)
        beta = 0.9999
        eff_fake = (1 - beta ** n_fake) / (1 - beta)
        eff_real = (1 - beta ** n_real) / (1 - beta)
        w_fake = 1 / eff_fake
        w_real = 1 / eff_real
        # Normalize
        total_w = w_fake + w_real
        w_fake = 2 * w_fake / total_w
        w_real = 2 * w_real / total_w
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return torch.tensor([w_fake, w_real], dtype=torch.float32)
