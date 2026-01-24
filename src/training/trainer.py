"""PyTorch Lightning module for deepfake detection GAN.

This module implements the training logic using PyTorch Lightning
for cleaner code and automatic handling of distributed training,
mixed precision, logging, and checkpointing.
"""

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.discriminator import DCTDiscriminator
from ..models.generator import UNetGenerator


class DeepfakeGANModule(L.LightningModule):
    """PyTorch Lightning module for Deepfake Detection GAN.
    
    Handles training of both discriminator and generator with adversarial
    training for improved robustness.
    
    Args:
        pretrained: Whether to use pretrained ConvNeXt weights.
        epsilon: Perturbation strength for generator.
        d_lr: Discriminator learning rate.
        g_lr: Generator learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        total_epochs: Total number of training epochs (for scheduler).
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        epsilon: float = 0.03,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        max_grad_norm: float = 1.0,
        total_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Disable automatic optimization for GAN training
        self.automatic_optimization = False
        
        # Models
        self.discriminator = DCTDiscriminator(pretrained=pretrained)
        self.generator = UNetGenerator()
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Hyperparameters
        self.epsilon = epsilon
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.max_grad_norm = max_grad_norm
        self.total_epochs = total_epochs
        
        # Validation outputs storage
        self._val_predictions = []
        self._val_labels = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.discriminator(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a single training step.
        
        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.
        """
        images, labels = batch
        
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
        
        # Create target labels
        ones = torch.ones(len(real_samples), 1, device=self.device)
        zeros = torch.zeros(len(fake_samples), 1, device=self.device)
        
        # ===============================
        # Train Discriminator
        # ===============================
        d_opt.zero_grad()
        
        # Forward pass on real images
        real_pred = self.discriminator(real_samples)
        d_loss_real = self.criterion(real_pred, ones)
        
        # Forward pass on fake images
        fake_pred = self.discriminator(fake_samples)
        d_loss_fake = self.criterion(fake_pred, zeros)
        
        # Generate adversarial images from real samples (detached)
        with torch.no_grad():
            adv_images, _ = self.generator(real_samples, self.epsilon)
        
        # Forward pass on adversarial images (should still be detected as real)
        adv_pred = self.discriminator(adv_images)
        d_loss_adv = self.criterion(adv_pred, ones)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake + 0.5 * d_loss_adv
        
        # Backward and step
        self.manual_backward(d_loss)
        self.clip_gradients(d_opt, gradient_clip_val=self.max_grad_norm)
        d_opt.step()
        
        # ===============================
        # Train Generator
        # ===============================
        g_opt.zero_grad()
        
        # Generate adversarial images
        adv_images, perturbation = self.generator(real_samples, self.epsilon)
        
        # Forward through discriminator
        adv_pred_for_g = self.discriminator(adv_images)
        
        # Generator tries to fool discriminator (predict as fake/zero)
        g_target_zeros = torch.zeros(len(real_samples), 1, device=self.device)
        g_loss_adv = self.criterion(adv_pred_for_g, g_target_zeros)
        
        # L1 regularization on perturbation (encourage small perturbations)
        g_loss_perturb = torch.mean(torch.abs(perturbation))
        
        # Total generator loss
        g_loss = g_loss_adv + 0.1 * g_loss_perturb
        
        # Backward and step
        self.manual_backward(g_loss)
        self.clip_gradients(g_opt, gradient_clip_val=self.max_grad_norm)
        g_opt.step()
        
        # Compute accuracies
        with torch.no_grad():
            d_acc_real = ((real_pred > 0).float() == ones).float().mean()
            d_acc_fake = ((fake_pred > 0).float() == zeros).float().mean()
            g_acc_adv = (adv_pred_for_g <= 0).float().mean()
        
        # Log metrics
        self.log_dict({
            'train/d_loss': d_loss,
            'train/g_loss': g_loss,
            'train/d_loss_real': d_loss_real,
            'train/d_loss_fake': d_loss_fake,
            'train/d_loss_adv': d_loss_adv,
            'train/g_loss_adv': g_loss_adv,
            'train/g_loss_perturb': g_loss_perturb,
            'train/d_acc_real': d_acc_real,
            'train/d_acc_fake': d_acc_fake,
            'train/g_acc_adv': g_acc_adv,
        }, prog_bar=True, on_step=True, on_epoch=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Perform a validation step.
        
        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.
        """
        images, labels = batch
        
        # Forward pass
        logits = self.discriminator(images)
        loss = self.criterion(logits, labels.unsqueeze(1))
        
        # Convert logits to predictions
        predictions = (logits > 0).float().squeeze()
        
        # Store for epoch-end metrics
        self._val_predictions.append(predictions)
        self._val_labels.append(labels)
        
        # Log loss
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at end of epoch."""
        if not self._val_predictions:
            return
            
        # Concatenate all predictions and labels
        all_preds = torch.cat(self._val_predictions)
        all_labels = torch.cat(self._val_labels)
        
        # Compute accuracy
        accuracy = (all_preds == all_labels).float().mean()
        
        # Compute precision, recall, F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Log metrics
        self.log_dict({
            'val/accuracy': accuracy,
            'val/precision': precision,
            'val/recall': recall,
            'val/f1': f1,
        }, prog_bar=True)
        
        # Clear storage
        self._val_predictions.clear()
        self._val_labels.clear()
    
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
