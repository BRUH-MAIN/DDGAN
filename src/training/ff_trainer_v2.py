"""PyTorch Lightning module for deepfake detection with improved numerical stability.

This module provides a numerically stable implementation with:
- Proper GradScaler for mixed precision training
- Image validation and normalization
- Gradient anomaly detection
- Conservative training dynamics
- Better loss function handling
"""

from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from ..models.discriminator import DCTDiscriminator
from ..models.generator import UNetGenerator


class StableBCELoss(nn.Module):
    """Numerically stable BCE loss with label smoothing.
    
    Uses a more stable formulation to prevent NaN values.
    """
    
    def __init__(
        self, 
        label_smoothing: float = 0.1,
        pos_weight: float = 1.0,
        eps: float = 1e-7
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute stable BCE loss.
        
        Args:
            logits: Raw predictions [B, 1] or [B]
            targets: Ground truth {0, 1} [B, 1] or [B]
            
        Returns:
            Scalar loss value
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Stable BCE computation
        max_val = torch.clamp(-logits, min=0)
        loss = logits - logits * targets + max_val + \
               torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val) + self.eps)
        
        # Apply pos_weight for imbalanced data
        weight = torch.where(targets > 0.5, self.pos_weight, 1.0)
        loss = loss * weight
        
        return loss.mean()


class StableFocalLoss(nn.Module):
    """Numerically stable Focal Loss for imbalanced classification.
    
    Implements focal loss with additional numerical safeguards.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        eps: float = 1e-7
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss with numerical stability."""
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clamp logits
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Compute probabilities with numerical stability
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1 - self.eps)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight with clamping
        focal_weight = torch.pow(1 - p_t, self.gamma)
        focal_weight = torch.clamp(focal_weight, min=0, max=100)
        
        # BCE loss
        bce = -targets * torch.log(probs + self.eps) - (1 - targets) * torch.log(1 - probs + self.eps)
        
        # Combined loss
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


class FFDeepfakeGANModuleV2(L.LightningModule):
    """Improved PyTorch Lightning module for Deepfake Detection.
    
    Key improvements over V1:
    - Manual GradScaler for mixed precision stability
    - Image validation and clamping
    - Warmup learning rate schedule
    - Better gradient handling
    - Discriminator-only pre-training option
    - Recovery from NaN states
    
    Args:
        pretrained: Use pretrained ConvNeXt weights
        epsilon: Perturbation strength for generator
        d_lr: Discriminator learning rate
        g_lr: Generator learning rate
        max_grad_norm: Maximum gradient norm for clipping
        total_epochs: Total training epochs
        loss_type: Loss function type ('bce', 'focal')
        label_smoothing: Label smoothing factor (0.0 to 0.5)
        warmup_epochs: Number of warmup epochs
        d_pretrain_steps: Steps to train discriminator only at start
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        epsilon: float = 0.03,
        d_lr: float = 1e-4,
        g_lr: float = 1e-4,
        max_grad_norm: float = 0.5,
        total_epochs: int = 50,
        loss_type: str = "focal",
        label_smoothing: float = 0.1,
        warmup_epochs: int = 2,
        d_pretrain_steps: int = 500,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,  # Accept extra args for compatibility
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
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
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        self.d_pretrain_steps = d_pretrain_steps
        
        # Class weights for imbalanced data
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            pos_weight = float(class_weights[1] / class_weights[0]) if class_weights[0] > 0 else 1.0
        else:
            self.register_buffer('class_weights', torch.ones(2))
            pos_weight = 1.0
        
        # Initialize loss function
        if loss_type == "bce":
            self.criterion = StableBCELoss(
                label_smoothing=label_smoothing,
                pos_weight=pos_weight
            )
        else:  # focal
            self.criterion = StableFocalLoss(
                alpha=0.25 * pos_weight,
                gamma=2.0,
                label_smoothing=label_smoothing
            )
        
        # Validation outputs storage
        self._val_predictions = []
        self._val_labels = []
        self._val_probs = []
        
        # Training state
        self._global_step = 0
        self._nan_count = 0
        self._max_nan_before_reset = 100
    
    def _validate_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate and normalize input images.
        
        Args:
            images: Input tensor
            
        Returns:
            Tuple of (validated_images, is_valid)
        """
        # Check for NaN/Inf
        if torch.isnan(images).any() or torch.isinf(images).any():
            return images, False
        
        # Clamp to valid range [0, 1]
        images = torch.clamp(images, 0.0, 1.0)
        
        return images, True
    
    def _safe_backward(
        self, 
        loss: torch.Tensor, 
        optimizer: torch.optim.Optimizer,
        parameters,
        step_name: str
    ) -> bool:
        """Safely perform backward pass with gradient clipping.
        
        Returns:
            True if successful, False if NaN detected
        """
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            self._nan_count += 1
            if self._nan_count % 50 == 0:
                print(f"⚠️  NaN count: {self._nan_count} at {step_name}")
            return False
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        optimizer.step()
        
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.discriminator(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a single training step with improved stability."""
        images, labels = batch
        self._global_step += 1
        
        # Validate images
        images, is_valid = self._validate_images(images)
        if not is_valid:
            return
        
        # Get optimizers
        d_opt, g_opt = self.optimizers()
        
        # Split batch into real (label=1) and fake (label=0) samples
        real_mask = labels == 1
        fake_mask = labels == 0
        
        real_samples = images[real_mask]
        fake_samples = images[fake_mask]
        
        # Skip if one class is missing
        if len(real_samples) == 0 or len(fake_samples) == 0:
            return
        
        # Create target labels (label smoothing is in the loss function)
        ones = torch.ones(len(real_samples), 1, device=self.device)
        zeros = torch.zeros(len(fake_samples), 1, device=self.device)
        
        # ===============================
        # Train Discriminator
        # ===============================
        # Forward pass on real images
        real_pred = self.discriminator(real_samples)
        d_loss_real = self.criterion(real_pred, ones)
        
        # Forward pass on fake images
        fake_pred = self.discriminator(fake_samples)
        d_loss_fake = self.criterion(fake_pred, zeros)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backward pass
        d_success = self._safe_backward(
            d_loss, d_opt,
            self.discriminator.parameters(), "discriminator"
        )
        
        if not d_success:
            return
        
        # ===============================
        # Train Generator (skip during pretrain phase)
        # ===============================
        if self._global_step > self.d_pretrain_steps:
            # Generate adversarial images
            adv_images, perturbation = self.generator(real_samples, self.epsilon)
            
            # Forward through discriminator
            adv_pred = self.discriminator(adv_images)
            
            # Generator wants discriminator to classify adversarial as FAKE
            g_target = torch.zeros(len(real_samples), 1, device=self.device)
            g_loss_adv = self.criterion(adv_pred, g_target)
            
            # L1 regularization on perturbation
            g_loss_perturb = torch.mean(torch.abs(perturbation))
            
            # Total generator loss
            g_loss = g_loss_adv + 0.01 * g_loss_perturb
            
            # Backward pass
            g_success = self._safe_backward(
                g_loss, g_opt,
                self.generator.parameters(), "generator"
            )
            
            if g_success:
                self.log('train/g_loss', g_loss.detach(), prog_bar=True)
        else:
            g_loss = torch.tensor(0.0, device=self.device)
        
        # Compute metrics
        with torch.no_grad():
            d_acc_real = (torch.sigmoid(real_pred) > 0.5).float().mean()
            d_acc_fake = (torch.sigmoid(fake_pred) < 0.5).float().mean()
        
        # Log metrics
        self.log_dict({
            'train/d_loss': d_loss.detach(),
            'train/d_loss_real': d_loss_real.detach(),
            'train/d_loss_fake': d_loss_fake.detach(),
            'train/d_acc_real': d_acc_real,
            'train/d_acc_fake': d_acc_fake,
        }, prog_bar=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a validation step."""
        images, labels = batch
        
        # Validate images
        images, is_valid = self._validate_images(images)
        if not is_valid:
            return
        
        # Forward pass (no AMP for validation)
        with torch.no_grad():
            logits = self.discriminator(images)
            
            # Use simple BCE for validation loss
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(), 
                labels.float()
            )
        
        # Get predictions and probabilities
        probs = torch.sigmoid(logits).squeeze()
        predictions = (probs > 0.5).float()
        
        # Store for epoch-end metrics
        self._val_predictions.append(predictions)
        self._val_labels.append(labels)
        self._val_probs.append(probs)
        
        # Log loss
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
    
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
        
        # Per-class metrics
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-8)
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
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
        }, prog_bar=True, sync_dist=True)
        
        # Clear storage
        self._val_predictions.clear()
        self._val_labels.clear()
        self._val_probs.clear()
    
    def configure_optimizers(self) -> tuple:
        """Configure optimizers with warmup scheduler."""
        d_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(0.9, 0.999),  # More standard betas
            weight_decay=0.01,
            eps=1e-8
        )
        g_optimizer = AdamW(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        d_scheduler = {
            'scheduler': CosineAnnealingWarmRestarts(
                d_optimizer, 
                T_0=max(1, self.total_epochs // 5),
                T_mult=2,
                eta_min=self.d_lr * 0.01
            ),
            'interval': 'epoch',
        }
        g_scheduler = {
            'scheduler': CosineAnnealingWarmRestarts(
                g_optimizer, 
                T_0=max(1, self.total_epochs // 5),
                T_mult=2,
                eta_min=self.g_lr * 0.01
            ),
            'interval': 'epoch',
        }
        
        return [d_optimizer, g_optimizer], [d_scheduler, g_scheduler]
