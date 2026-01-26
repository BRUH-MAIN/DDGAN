"""PyTorch Lightning module for deepfake detection with imbalance handling.

This module extends the base trainer with:
- Multiple loss function options (Focal, AAML, Weighted BCE)
- Class weight support
- Improved metrics for imbalanced data
"""

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.discriminator import DCTDiscriminator
from ..models.generator import UNetGenerator
from .imbalance_losses import (
    FocalLoss, 
    WeightedBCELoss, 
    AAMLoss,
    AsymmetricFocalLoss,
    CombinedImbalanceLoss,
)


class FFDeepfakeGANModule(L.LightningModule):
    """PyTorch Lightning module for Deepfake Detection with imbalance handling.
    
    Supports multiple loss functions designed for imbalanced data:
    - bce: Standard binary cross-entropy
    - focal: Focal loss (focuses on hard examples)
    - weighted_bce: Class-weighted BCE
    - aaml: Additive Angular Margin Loss
    - combined: Combination of focal + AAML
    
    Args:
        pretrained: Use pretrained ConvNeXt weights
        epsilon: Perturbation strength for generator
        d_lr: Discriminator learning rate
        g_lr: Generator learning rate
        max_grad_norm: Maximum gradient norm for clipping
        total_epochs: Total training epochs
        loss_type: Loss function type
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss
        aaml_margin: Margin for AAML
        aaml_scale: Scale for AAML
        class_weights: Optional class weights [fake_w, real_w]
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        epsilon: float = 0.03,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        max_grad_norm: float = 1.0,
        total_epochs: int = 50,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        aaml_margin: float = 0.5,
        aaml_scale: float = 30.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Disable automatic optimization for GAN training
        self.automatic_optimization = False
        
        # Models
        self.discriminator = DCTDiscriminator(pretrained=pretrained)
        self.generator = UNetGenerator()
        
        # Store hyperparameters
        self.epsilon = epsilon
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.max_grad_norm = max_grad_norm
        self.total_epochs = total_epochs
        self.loss_type = loss_type
        
        # Class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(2))
        
        # Initialize loss functions
        self._init_loss_functions(
            loss_type, focal_gamma, focal_alpha, 
            aaml_margin, aaml_scale, class_weights
        )
        
        # Validation outputs storage
        self._val_predictions = []
        self._val_labels = []
        self._val_probs = []  # Store probabilities for AUC
    
    def _init_loss_functions(
        self,
        loss_type: str,
        focal_gamma: float,
        focal_alpha: float,
        aaml_margin: float,
        aaml_scale: float,
        class_weights: Optional[torch.Tensor],
    ):
        """Initialize the appropriate loss function."""
        # Compute pos_weight from class_weights if available
        # class_weights = [fake_w, real_w], pos_weight is for real (positive) class
        pos_weight = 1.0
        if class_weights is not None:
            # Real (label=1) is minority, so give it higher weight
            pos_weight = class_weights[1].item() / class_weights[0].item()
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "focal":
            # For focal loss, alpha should be higher for minority class
            # If real is minority, increase alpha
            adjusted_alpha = min(0.75, focal_alpha * pos_weight) if pos_weight > 1 else focal_alpha
            self.criterion = FocalLoss(alpha=adjusted_alpha, gamma=focal_gamma)
        elif loss_type == "weighted_bce":
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        elif loss_type == "asymmetric_focal":
            self.criterion = AsymmetricFocalLoss(
                gamma_pos=0.0,  # Less focus on easy positives
                gamma_neg=focal_gamma,  # More focus on hard negatives
                alpha=focal_alpha * pos_weight if pos_weight > 1 else focal_alpha,
            )
        elif loss_type == "aaml":
            # AAML needs feature dimension from discriminator
            # Assuming discriminator outputs 512-dim features before final layer
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.aaml_loss = AAMLoss(
                in_features=512,  # Match discriminator feature dim
                scale=aaml_scale,
                margin=aaml_margin,
                class_weights=class_weights,
            )
        elif loss_type == "combined":
            self.criterion = CombinedImbalanceLoss(
                in_features=512,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                aaml_scale=aaml_scale,
                aaml_margin=aaml_margin,
                aaml_weight=0.3,  # 30% AAML, 70% focal
                class_weights=class_weights,
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.discriminator(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a single training step."""
        images, labels = batch
        
        # Ensure images are in valid range [0, 1] and check for NaN
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"⚠️  Warning: NaN/Inf detected in input images at batch {batch_idx}")
            return
        
        # Get optimizers
        d_opt, g_opt = self.optimizers()
        
        # Split batch into real (label=1) and fake (label=0) samples
        real_mask = labels == 1
        fake_mask = labels == 0
        
        real_samples = images[real_mask]
        fake_samples = images[fake_mask]
        
        # Skip if one class is missing (shouldn't happen with weighted sampling)
        if len(real_samples) == 0 or len(fake_samples) == 0:
            return
        
        # Create target labels with label smoothing for stability
        # Real: 0.9 instead of 1.0, Fake: 0.1 instead of 0.0
        ones = torch.ones(len(real_samples), 1, device=self.device) * 0.9
        zeros = torch.zeros(len(fake_samples), 1, device=self.device) + 0.1
        
        # ===============================
        # Train Discriminator
        # ===============================
        d_opt.zero_grad()
        
        # Forward pass on real images
        real_pred = self.discriminator(real_samples)
        d_loss_real = self._compute_loss(real_pred, ones)
        
        # Forward pass on fake images
        fake_pred = self.discriminator(fake_samples)
        d_loss_fake = self._compute_loss(fake_pred, zeros)
        
        # Check for NaN in loss values
        if torch.isnan(d_loss_real) or torch.isnan(d_loss_fake):
            print(f"⚠️  Warning: NaN detected in discriminator loss at batch {batch_idx}")
            print(f"   d_loss_real: {d_loss_real}, d_loss_fake: {d_loss_fake}")
            d_opt.zero_grad()  # Clear gradients
            return
        
        # Generate adversarial images from real samples
        with torch.no_grad():
            adv_images, _ = self.generator(real_samples, self.epsilon)
        
        # Forward pass on adversarial images
        adv_pred = self.discriminator(adv_images)
        d_loss_adv = self._compute_loss(adv_pred, ones)
        
        # Check for NaN
        if torch.isnan(d_loss_adv):
            print(f"⚠️  Warning: NaN detected in adversarial loss at batch {batch_idx}")
            d_opt.zero_grad()
            return
        
        # Total discriminator loss with smaller weights for stability
        d_loss = 0.8 * (d_loss_real + d_loss_fake) + 0.2 * d_loss_adv
        
        # Clamp loss to prevent extreme values
        d_loss = torch.clamp(d_loss, min=-100, max=100)
        
        # Backward and step
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.max_grad_norm)
        d_opt.step()
        
        # ===============================
        # Train Generator
        # ===============================
        g_opt.zero_grad()
        
        # Generate adversarial images
        adv_images, perturbation = self.generator(real_samples, self.epsilon)
        
        # Check perturbation
        if torch.isnan(perturbation).any() or torch.isinf(perturbation).any():
            print(f"⚠️  Warning: NaN/Inf in perturbation at batch {batch_idx}")
            g_opt.zero_grad()
            return
        
        # Forward through discriminator
        adv_pred_for_g = self.discriminator(adv_images)
        
        # Generator tries to fool discriminator (wants logits < 0, i.e., fake classification)
        g_target_ones = torch.ones(len(real_samples), 1, device=self.device) * 0.1  # Target fake
        g_loss_adv = self._compute_loss(adv_pred_for_g, g_target_ones)
        
        # L1 regularization on perturbation
        g_loss_perturb = torch.mean(torch.abs(perturbation))
        
        # Total generator loss
        g_loss = g_loss_adv + 0.01 * g_loss_perturb  # Reduced perturbation weight
        
        # Clamp loss
        g_loss = torch.clamp(g_loss, min=-100, max=100)
        
        # Check for NaN
        if torch.isnan(g_loss):
            print(f"⚠️  Warning: NaN detected in generator loss at batch {batch_idx}")
            g_opt.zero_grad()
            return
        
        # Backward and step
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.max_grad_norm)
        g_opt.step()
        
        # Compute metrics
        with torch.no_grad():
            d_acc_real = ((real_pred > 0).float() == (ones > 0.5).float()).float().mean()
            d_acc_fake = ((fake_pred > 0).float() == (zeros > 0.5).float()).float().mean()
            g_acc_adv = (adv_pred_for_g < 0).float().mean()
        
        # Log metrics
        self.log_dict({
            'train/d_loss_step': d_loss.detach(),
            'train/g_loss_step': g_loss.detach(),
            'train/d_loss_real_step': d_loss_real.detach(),
            'train/d_loss_fake_step': d_loss_fake.detach(),
            'train/d_acc_real_step': d_acc_real,
            'train/d_acc_fake_step': d_acc_fake,
            'train/g_acc_adv_step': g_acc_adv,
        }, prog_bar=True, on_step=True, on_epoch=False)
    
    def _compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss based on the configured loss type."""
        if self.loss_type == "combined":
            loss_dict = self.criterion(logits, targets, features)
            return loss_dict['total']
        elif self.loss_type == "aaml" and hasattr(self, 'aaml_loss') and features is not None:
            focal = self.criterion(logits, targets)
            aaml, _ = self.aaml_loss(features, targets)
            return 0.7 * focal + 0.3 * aaml
        else:
            return self.criterion(logits, targets)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a validation step."""
        images, labels = batch
        
        # Forward pass
        logits = self.discriminator(images)
        loss = self._compute_loss(logits, labels.unsqueeze(1))
        
        # Get predictions and probabilities
        probs = torch.sigmoid(logits).squeeze()
        predictions = (logits > 0).float().squeeze()
        
        # Store for epoch-end metrics
        self._val_predictions.append(predictions)
        self._val_labels.append(labels)
        self._val_probs.append(probs)
        
        # Log loss
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at end of epoch."""
        if not self._val_predictions:
            return
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(self._val_predictions)
        all_labels = torch.cat(self._val_labels)
        all_probs = torch.cat(self._val_probs)
        
        # Basic metrics
        accuracy = (all_preds == all_labels).float().mean()
        
        # Per-class metrics (important for imbalanced data)
        # True/False Positives/Negatives
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Specificity (true negative rate) - important for detecting fakes
        specificity = tn / (tn + fp + 1e-8)
        
        # Balanced accuracy (important for imbalanced data)
        balanced_acc = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient (robust metric for imbalanced data)
        mcc_num = tp * tn - fp * fn
        mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
        mcc = mcc_num / mcc_den
        
        # Log metrics
        self.log_dict({
            'val/accuracy': accuracy,
            'val/balanced_accuracy': balanced_acc,
            'val/precision': precision,
            'val/recall': recall,
            'val/specificity': specificity,
            'val/f1': f1,
            'val/mcc': mcc,
        }, prog_bar=True)
        
        # Clear storage
        self._val_predictions.clear()
        self._val_labels.clear()
        self._val_probs.clear()
    
    def configure_optimizers(self) -> tuple:
        """Configure optimizers and schedulers."""
        d_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        g_optimizer = AdamW(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        
        d_scheduler = {
            'scheduler': CosineAnnealingLR(d_optimizer, T_max=self.total_epochs),
            'interval': 'epoch',
        }
        g_scheduler = {
            'scheduler': CosineAnnealingLR(g_optimizer, T_max=self.total_epochs),
            'interval': 'epoch',
        }
        
        return [d_optimizer, g_optimizer], [d_scheduler, g_scheduler]
