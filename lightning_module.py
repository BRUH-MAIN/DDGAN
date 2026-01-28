"""
PyTorch Lightning Training Module for Deepfake Detection GAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from models import Discriminator, Generator


class DeepfakeGAN(pl.LightningModule):
    """
    PyTorch Lightning module for adversarial robustness training
    
    Training Philosophy:
    - NOT traditional GAN training (generator doesn't create realistic fakes)
    - Generator creates adversarial perturbations to test discriminator
    - Discriminator learns to be robust against perturbations
    - Goal: Discriminator that generalizes well to unseen deepfakes
    
    Uses manual optimization for proper alternating D/G training.
    """
    
    # Enable manual optimization for proper GAN training
    automatic_optimization = False
    
    def __init__(
        self,
        # Model config
        d_backbone='convnext_tiny',
        d_pretrained=True,
        g_base_channels=64,
        epsilon=0.03,
        
        # Training config
        lr=2e-4,
        betas=(0.5, 0.999),
        weight_decay=0.01,
        consistency_weight=1.0,  # Weight for consistency loss in discriminator
        margin=1.0,  # Margin for generator margin loss
        perturb_weight=0.1,
        scheduler_type='cosine',
        max_epochs=50
    ):
        """
        Initialize DeepfakeGAN module
        
        Args:
            d_backbone: Discriminator backbone architecture
            d_pretrained: Use pretrained weights for discriminator
            g_base_channels: Base channels for generator
            epsilon: Maximum perturbation magnitude
            lr: Learning rate
            betas: Adam beta parameters
            weight_decay: Weight decay for AdamW
            consistency_weight: Weight for consistency loss (prediction drift penalty)
            margin: Margin for generator margin loss
            perturb_weight: Weight for perturbation regularization
            scheduler_type: Learning rate scheduler type
            max_epochs: Maximum training epochs
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize models
        self.discriminator = Discriminator(
            backbone=d_backbone,
            pretrained=d_pretrained,
            num_classes=1
        )
        
        self.generator = Generator(
            input_channels=3,
            base_channels=g_base_channels,
            epsilon=epsilon
        )
        
        # Loss function with class weighting for imbalanced data
        # Dataset is ~88% fake, ~12% real, so weight real samples higher
        # pos_weight > 1 increases recall for positive class (real=1 in BCE)
        # Using sqrt of ratio for less aggressive weighting
        self.register_buffer('pos_weight', torch.tensor([3.0]))  # sqrt(88/12) ≈ 2.7
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Metrics storage
        self.validation_step_outputs = []
    
    def forward(self, x):
        """Forward pass through discriminator"""
        return self.discriminator(x)
    
    def consistency_loss(self, logits_clean, logits_adv):
        """
        Penalize prediction drift between clean and adversarial inputs.
        Operates in probability space to preserve ranking.
        """
        p_clean = torch.sigmoid(logits_clean)
        p_adv = torch.sigmoid(logits_adv)
        return torch.mean((p_clean - p_adv) ** 2)

    def generator_margin_loss(self, logits, labels, margin=1.0):
        """
        Margin-based adversarial loss.
        Encourages confidence reduction without forcing label flip.
        labels: 1 for real, 0 for fake
        """
        signed_logits = (2 * labels - 1) * logits
        return torch.mean(F.relu(margin - signed_logits))
    
    def training_step(self, batch, batch_idx):
        """
        Training step following the adversarial robustness strategy
        
        Step 1: Separate real and fake samples
        Step 2: Train discriminator (classify correctly + be robust)
        Step 3: Train generator (create effective perturbations)
        """
        images, labels = batch
        batch_size = images.size(0)
        
        # Check for NaN/inf in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"[WARNING] batch={batch_idx}: NaN/inf detected in input images!")
        
        # Separate real and fake images
        # IMPORTANT: Verify label convention from dataset!
        # HuggingFace celebdfv2: 0='real', 1='fake' (index-based from folder order)
        real_mask = labels == 0  # 0 = real
        fake_mask = labels == 1  # 1 = fake
        
        real_images = images[real_mask]
        fake_images = images[fake_mask]
        
        # Handle edge cases where batch might not have both classes
        num_real = real_images.size(0)
        num_fake = fake_images.size(0)
        
        # Initialize losses (with requires_grad=True for backward compatibility)
        loss_real = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fake = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_consistency = torch.tensor(0.0, device=self.device, requires_grad=True)
        g_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        g_loss_margin = torch.tensor(0.0, device=self.device, requires_grad=True)
        g_loss_perturb = torch.tensor(0.0, device=self.device, requires_grad=True)
        d_acc_real = torch.tensor(0.5, device=self.device)
        d_acc_fake = torch.tensor(0.5, device=self.device)
        g_acc_adv = torch.tensor(0.5, device=self.device)  # Mean confidence on adversarial images
        
        # ===== TRAIN DISCRIMINATOR =====
        # 1. Forward real images
        if num_real > 0:
            real_labels = torch.ones(num_real, 1, device=self.device)
            real_outputs = self.discriminator(real_images)
            loss_real = self.criterion(real_outputs, real_labels)
            with torch.no_grad():
                d_acc_real = ((real_outputs > 0).float() == real_labels).float().mean()
        
        # 2. Forward fake images
        if num_fake > 0:
            fake_labels = torch.zeros(num_fake, 1, device=self.device)
            fake_outputs = self.discriminator(fake_images)
            loss_fake = self.criterion(fake_outputs, fake_labels)
            with torch.no_grad():
                d_acc_fake = ((fake_outputs > 0).float() == fake_labels).float().mean()
        
        # 3. Generate adversarial images from real images and compute consistency loss
        loss_consistency = torch.tensor(0.0, device=self.device, requires_grad=True)
        if num_real > 0:
            adv_images, perturbation = self.generator(real_images)
            adv_logits = self.discriminator(adv_images.detach())
            
            # Consistency loss: penalize prediction drift (NOT BCE on adversarial labels)
            loss_consistency = self.consistency_loss(real_outputs.detach(), adv_logits)
        
        # Combined discriminator loss
        # Warmup: skip consistency loss for first 3 epochs to let G learn to attack first
        if self.current_epoch < 3:
            d_loss = loss_real + loss_fake
        else:
            d_loss = loss_real + loss_fake + self.hparams.consistency_weight * loss_consistency
        
        # ===== MANUAL OPTIMIZATION =====
        # Get optimizers
        d_opt, g_opt = self.optimizers()
        
        # Step 1: Update Discriminator FIRST (before computing g_loss)
        # This avoids the inplace modification error
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.clip_gradients(d_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        d_opt.step()
        
        # ===== TRAIN GENERATOR =====
        # Generate fresh adversarial images AFTER discriminator update
        # This creates a new computation graph with updated discriminator weights
        g_loss_margin = torch.tensor(0.0, device=self.device, requires_grad=True)
        if num_real > 0:
            # Generate adversarial images (fresh forward pass)
            adv_images_g, perturbation_g = self.generator(real_images)
            
            # Forward through discriminator (no detach!)
            adv_logits_g = self.discriminator(adv_images_g)
            
            # Margin-based loss (reduce confidence without forcing label flip)
            labels_real = torch.ones_like(adv_logits_g)
            g_loss_margin = self.generator_margin_loss(
                adv_logits_g,
                labels_real,
                margin=self.hparams.margin
            )
            
            # Perturbation regularization (L1 norm)
            g_loss_perturb = torch.mean(torch.abs(perturbation_g))
            
            # Combined generator loss
            g_loss = g_loss_margin + self.hparams.perturb_weight * g_loss_perturb
            
            # Generator effectiveness: how much confidence was reduced
            with torch.no_grad():
                g_acc_adv = torch.sigmoid(adv_logits_g).mean()  # Lower = more effective
        
        # Step 2: Update Generator (only if we have real images to generate adversarial samples)
        # Must include backward AND step together for AMP scaler compatibility
        if num_real > 0:
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_gradients(g_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            g_opt.step()
        
        # Log metrics
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/d_acc_real', d_acc_real, on_step=False, on_epoch=True)
        self.log('train/d_acc_fake', d_acc_fake, on_step=False, on_epoch=True)
        self.log('train/g_confidence_reduction', g_acc_adv, on_step=False, on_epoch=True)
        self.log('train/loss_real', loss_real, on_step=False, on_epoch=True)
        self.log('train/loss_fake', loss_fake, on_step=False, on_epoch=True)
        self.log('train/loss_consistency', loss_consistency, on_step=True, on_epoch=True)
        self.log('train/g_loss_margin', g_loss_margin, on_step=True, on_epoch=True)
        
        # NaN check
        if torch.isnan(d_loss) or torch.isnan(g_loss):
            print(f'[ERROR] NaN loss at batch {batch_idx}!')
            print(f'  d_loss={d_loss}, g_loss={g_loss}')
    
    def on_train_epoch_end(self):
        """Step learning rate schedulers at end of epoch"""
        d_sch, g_sch = self.lr_schedulers()
        d_sch.step()
        g_sch.step()
    
    def validation_step(self, batch, batch_idx):
        """Validation step with adversarial evaluation"""
        images, labels = batch
        
        # Forward pass on clean images
        logits_clean = self.discriminator(images)
        probs_clean = torch.sigmoid(logits_clean)
        preds = (probs_clean > 0.5).long().squeeze()
        
        # Adversarial evaluation (NO gradients)
        with torch.no_grad():
            adv_images, _ = self.generator(images)
            logits_adv = self.discriminator(adv_images)
            probs_adv = torch.sigmoid(logits_adv)
        
        # Store for epoch-level metrics
        self.validation_step_outputs.append({
            'labels': labels.cpu(),
            'preds': preds.cpu(),
            'probs_clean': probs_clean.cpu().squeeze(),
            'probs_adv': probs_adv.cpu().squeeze()
        })
        
        # Calculate loss
        labels_float = labels.float().unsqueeze(1)
        loss = self.criterion(logits_clean, labels_float)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate and log epoch-level metrics including adversarial robustness"""
        # Gather all predictions and labels
        all_labels = []
        all_preds = []
        all_probs_clean = []
        all_probs_adv = []
        
        for output in self.validation_step_outputs:
            all_labels.append(output['labels'])
            all_preds.append(output['preds'])
            all_probs_clean.append(output['probs_clean'])
            all_probs_adv.append(output['probs_adv'])
        
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs_clean = torch.cat(all_probs_clean).numpy()
        all_probs_adv = torch.cat(all_probs_adv).numpy()
        
        # Calculate standard metrics (on clean images)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Calculate AUC for clean and adversarial
        try:
            auc_clean = roc_auc_score(all_labels, all_probs_clean)
        except:
            auc_clean = 0.0
        
        try:
            auc_adv = roc_auc_score(all_labels, all_probs_adv)
        except:
            auc_adv = 0.0
        
        # Robustness gap (lower is better)
        robustness_gap = auc_clean - auc_adv
        
        # Log metrics
        self.log('val/accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('val/precision', precision, on_epoch=True)
        self.log('val/recall', recall, on_epoch=True)
        self.log('val/f1', f1, on_epoch=True)
        self.log('val/auc_clean', auc_clean, on_epoch=True, prog_bar=True)
        self.log('val/auc_adv', auc_adv, on_epoch=True, prog_bar=True)
        self.log('val/robustness_gap', robustness_gap, on_epoch=True, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
        
        print(f"\n[Validation] Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
              f"Rec: {recall:.4f} | F1: {f1:.4f}")
        print(f"[Robustness] AUC_clean: {auc_clean:.4f} | AUC_adv: {auc_adv:.4f} | "
              f"Gap: {robustness_gap:.4f}")
    
    def configure_optimizers(self):
        """Configure separate optimizers for discriminator and generator"""
        # Discriminator optimizer
        d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )
        
        # Generator optimizer (slightly lower lr for stability)
        g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.lr * 0.5,  # Generator learns slower
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate schedulers
        if self.hparams.scheduler_type == 'cosine':
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                d_optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6
            )
            g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                g_optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6
            )
        else:
            d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.5)
            g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.5)
        
        return (
            [d_optimizer, g_optimizer],
            [d_scheduler, g_scheduler]
        )


if __name__ == "__main__":
    # Test DeepfakeGAN module
    print("Testing DeepfakeGAN module...")
    
    model = DeepfakeGAN(
        d_backbone='convnext_tiny',
        d_pretrained=False,  # Faster for testing
        g_base_channels=64,
        epsilon=0.03
    )
    
    # Create dummy batch
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nBatch size: {batch_size}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: Real={torch.sum(labels == 0).item()}, Fake={torch.sum(labels == 1).item()}")
    
    # Test training step
    model.training_step((images, labels), 0)
    
    print("\n✓ DeepfakeGAN module test passed!")
