"""GAN training module for deepfake detection.

This module implements the training loop for the adversarial deepfake
detection system, including discriminator and generator training.
"""

import os
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..utils.metrics import compute_accuracy, compute_precision_recall_f1


class DeepfakeGANTrainer:
    """Trainer for the adversarial deepfake detection GAN.
    
    Handles training of both discriminator and generator with adversarial
    training for improved robustness.
    
    Args:
        discriminator: The discriminator model.
        generator: The generator model.
        device: Torch device for training.
        d_lr: Discriminator learning rate (default: 2e-4).
        g_lr: Generator learning rate (default: 2e-4).
        use_amp: Whether to use automatic mixed precision (default: True).
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0).
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        device: torch.device,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        use_amp: bool = True,
        max_grad_norm: float = 1.0
    ) -> None:
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.max_grad_norm = max_grad_norm
        
        # Move models to device
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)
        
        # Compile models for faster execution (PyTorch 2.0+)
        try:
            self.discriminator = torch.compile(self.discriminator)
            self.generator = torch.compile(self.generator)
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}. Using eager mode.")
        
        # Optimizers with AdamW
        self.d_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        self.g_optimizer = AdamW(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        
        # Loss function (BCEWithLogitsLoss includes sigmoid)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate schedulers
        self.d_scheduler: Optional[CosineAnnealingLR] = None
        self.g_scheduler: Optional[CosineAnnealingLR] = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
    
    def setup_schedulers(self, total_epochs: int) -> None:
        """Setup learning rate schedulers.
        
        Args:
            total_epochs: Total number of training epochs.
        """
        self.d_scheduler = CosineAnnealingLR(self.d_optimizer, T_max=total_epochs)
        self.g_scheduler = CosineAnnealingLR(self.g_optimizer, T_max=total_epochs)
    
    def train_step(
        self, 
        real_images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Optional[Dict[str, float]]:
        """Perform a single training step.
        
        Args:
            real_images: Batch of images [B, 3, 224, 224].
            labels: Binary labels (1 for real, 0 for fake).
            
        Returns:
            Dictionary of metrics or None if batch is invalid.
        """
        # Split batch into real (label=1) and fake (label=0) samples
        real_mask = labels == 1
        fake_mask = labels == 0
        
        real_samples = real_images[real_mask]
        fake_samples = real_images[fake_mask]
        
        # Handle edge cases where one class is missing
        if len(real_samples) == 0 or len(fake_samples) == 0:
            return None
        
        # Create target labels
        ones = torch.ones(len(real_samples), 1, device=self.device)
        zeros = torch.zeros(len(fake_samples), 1, device=self.device)
        
        # ===============================
        # Train Discriminator
        # ===============================
        self.d_optimizer.zero_grad()
        
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            # Forward pass on real images
            real_pred = self.discriminator(real_samples)
            d_loss_real = self.criterion(real_pred, ones)
            
            # Forward pass on fake images
            fake_pred = self.discriminator(fake_samples)
            d_loss_fake = self.criterion(fake_pred, zeros)
            
            # Generate adversarial images from real samples (detached)
            with torch.no_grad():
                adv_images, _ = self.generator(real_samples)
            
            # Forward pass on adversarial images (should still be detected as real)
            adv_pred = self.discriminator(adv_images)
            d_loss_adv = self.criterion(adv_pred, ones)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + 0.5 * d_loss_adv
        
        # Backward pass for discriminator
        if self.use_amp:
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.d_optimizer)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.scaler.step(self.d_optimizer)
        else:
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.d_optimizer.step()
        
        # ===============================
        # Train Generator
        # ===============================
        self.g_optimizer.zero_grad()
        
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            # Generate adversarial images
            adv_images, perturbation = self.generator(real_samples)
            
            # Forward through discriminator
            adv_pred_for_g = self.discriminator(adv_images)
            
            # Generator tries to fool discriminator (predict as fake/zero)
            g_loss_adv = self.criterion(adv_pred_for_g, zeros)
            
            # L1 regularization on perturbation (encourage small perturbations)
            g_loss_perturb = torch.mean(torch.abs(perturbation))
            
            # Total generator loss
            g_loss = g_loss_adv + 0.1 * g_loss_perturb
        
        # Backward pass for generator
        if self.use_amp:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.g_optimizer)
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
        else:
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
            self.g_optimizer.step()
        
        # Compute accuracies
        with torch.no_grad():
            d_acc_real = ((real_pred > 0).float() == ones).float().mean().item()
            d_acc_fake = ((fake_pred > 0).float() == zeros).float().mean().item()
            g_acc_adv = ((adv_pred_for_g <= 0).float()).mean().item()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_loss_adv': d_loss_adv.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_perturb': g_loss_perturb.item(),
            'd_acc_real': d_acc_real,
            'd_acc_fake': d_acc_fake,
            'g_acc_adv': g_acc_adv
        }
    
    def validate(
        self, 
        val_loader: DataLoader, 
        gpu_transform: nn.Module
    ) -> Dict[str, float]:
        """Validate the discriminator on validation set.
        
        Args:
            val_loader: Validation DataLoader.
            gpu_transform: GPU transforms to apply.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.discriminator.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Apply GPU transforms
                images = gpu_transform(images)
                
                # Forward pass
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.discriminator(images)
                    loss = self.criterion(logits, labels.unsqueeze(1))
                
                total_loss += loss.item()
                num_batches += 1
                
                # Convert logits to predictions
                predictions = (logits > 0).float().squeeze()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        self.discriminator.train()
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        accuracy = compute_accuracy(all_predictions, all_labels)
        metrics_dict = compute_precision_recall_f1(all_predictions, all_labels)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': accuracy,
            'val_precision': metrics_dict['precision'],
            'val_recall': metrics_dict['recall'],
            'val_f1': metrics_dict['f1']
        }
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int, 
        metrics: Dict[str, float]
    ) -> None:
        """Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
        }
        
        if self.d_scheduler is not None:
            checkpoint['d_scheduler_state_dict'] = self.d_scheduler.state_dict()
        if self.g_scheduler is not None:
            checkpoint['g_scheduler_state_dict'] = self.g_scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        """Load a training checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            
        Returns:
            Tuple of (epoch, metrics) from the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        
        if 'd_scheduler_state_dict' in checkpoint and self.d_scheduler is not None:
            self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        if 'g_scheduler_state_dict' in checkpoint and self.g_scheduler is not None:
            self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def step_schedulers(self) -> None:
        """Step both learning rate schedulers."""
        if self.d_scheduler is not None:
            self.d_scheduler.step()
        if self.g_scheduler is not None:
            self.g_scheduler.step()
