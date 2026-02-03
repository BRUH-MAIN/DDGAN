"""
PyTorch Lightning Training Module for Deepfake Detection GAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from models import Discriminator, DualStreamDiscriminator, create_discriminator, Generator


class DeepfakeGAN(pl.LightningModule):
    """
    PyTorch Lightning module for adversarial robustness training
    
    Training Philosophy:
    - NOT traditional GAN training (generator doesn't create realistic fakes)
    - Generator creates adversarial perturbations to test discriminator
    - Discriminator learns to be robust against perturbations
    - Goal: Discriminator that generalizes well to unseen deepfakes
    
    Supports two discriminator architectures:
    - single_stream: Legacy DCT-only discriminator (grayscale DCT)
    - dual_stream: RGB + DCT fusion with learnable frequency filters
    
    Uses manual optimization for proper alternating D/G training.
    """
    
    # Enable manual optimization for proper GAN training
    automatic_optimization = False
    
    def __init__(
        self,
        # Model config
        d_backbone='convnext_tiny',
        d_pretrained=True,
        d_type='dual_stream',  # 'single_stream' or 'dual_stream'
        d_fusion_type='concat',  # For dual_stream: 'concat' or 'add'
        g_base_channels=64,
        epsilon=0.03,
        
        # Training config
        lr=2e-4,
        d_lr_mult=0.2,
        g_lr_mult=0.5,
        betas=(0.5, 0.999),
        weight_decay=0.01,
        consistency_weight=1.0,  # Weight for consistency loss in discriminator
        margin_max=0.3,  # Max margin for generator loss (warmup to this value)
        perturb_weight=0.1,
        scheduler_type='cosine',
        max_epochs=50
    ):
        """
        Initialize DeepfakeGAN module
        
        Args:
            d_backbone: Discriminator backbone architecture
            d_pretrained: Use pretrained weights for discriminator
            d_type: Discriminator type ('single_stream' or 'dual_stream')
            d_fusion_type: Fusion type for dual_stream ('concat' or 'add')
            g_base_channels: Base channels for generator
            epsilon: Maximum perturbation magnitude
            lr: Base learning rate
            d_lr_mult: Discriminator LR multiplier
            g_lr_mult: Generator LR multiplier
            betas: Adam beta parameters
            weight_decay: Weight decay for AdamW
            consistency_weight: Weight for consistency loss (prediction drift penalty)
            margin_max: Max margin for generator loss (warmup to this value)
            perturb_weight: Weight for perturbation regularization
            scheduler_type: Learning rate scheduler type
            max_epochs: Maximum training epochs
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize discriminator using factory function
        self.discriminator = create_discriminator(
            discriminator_type=d_type,
            backbone=d_backbone,
            pretrained=d_pretrained,
            num_classes=1,
            fusion_type=d_fusion_type
        )
        
        self.generator = Generator(
            input_channels=3,
            base_channels=g_base_channels,
            epsilon=epsilon
        )
        
        # Loss function with class weighting for imbalanced data
        # Label convention: real=1 (positive), fake=0 (negative)
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

    def generator_margin_loss(self, logits, margin):
        """
        Margin-based generator loss for adversarial robustness.

        Only pushes logits DOWN until they cross below the margin threshold.
        Once logit ≤ margin, generator has succeeded → no more pressure.

        Args:
            logits: Discriminator logits for adversarial real images [B, 1]
            margin: Current margin threshold (warmup)

        Returns:
            Scalar loss (mean of margin violations)
        """
        violation = F.relu(logits - margin)
        return torch.mean(violation)

    def current_margin(self):
        """Warmup margin schedule: m(t) = m_max * min(1, t / 5)."""
        t = float(self.current_epoch + 1)
        return self.hparams.margin_max * min(1.0, t / 5.0)
    
    def perturbation_loss(self, perturbation):
        """
        Scale-invariant perturbation regularization (mean squared magnitude).
        
        Args:
            perturbation: Perturbation tensor [B, C, H, W]
            
        Returns:
            Scalar loss (mean squared magnitude)
        """
        # Mean squared magnitude makes the penalty independent of image size
        return torch.mean(perturbation ** 2)
    
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
        # Label convention (normalized in DataModule): real=1, fake=0
        real_mask = labels == 1  # 1 = real
        fake_mask = labels == 0  # 0 = fake
        
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
        g_loss_confidence = torch.tensor(0.0, device=self.device, requires_grad=True)
        g_loss_perturb = torch.tensor(0.0, device=self.device, requires_grad=True)
        d_acc_real = torch.tensor(0.5, device=self.device)
        d_acc_fake = torch.tensor(0.5, device=self.device)
        adv_logit_mean = torch.tensor(0.0, device=self.device)  # Track mean adversarial logit
        margin_violation_rate = torch.tensor(0.0, device=self.device)
        margin_violation_mean = torch.tensor(0.0, device=self.device)
        perturb_energy = torch.tensor(0.0, device=self.device)
        
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
        # Progressive consistency: ramp up gradually over first 5 epochs (recovery slope, not cliff)
        cons_w = min(1.0, self.current_epoch / 5)
        d_loss = loss_real + loss_fake + cons_w * self.hparams.consistency_weight * loss_consistency
        
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
            
            # Margin warmup schedule
            margin_t = self.current_margin()
            margin_tensor = torch.tensor(margin_t, device=self.device, dtype=adv_logits_g.dtype)
            
            # Margin-based loss: only push logits below margin, then stop
            g_loss_confidence = self.generator_margin_loss(adv_logits_g, margin_tensor)
            g_loss_margin = g_loss_confidence
            
            # Perturbation regularization (L2 norm - smoother gradients than L1)
            g_loss_perturb = self.perturbation_loss(perturbation_g)
            perturb_energy = torch.mean(perturbation_g ** 2)
            
            # Combined generator loss
            g_loss = g_loss_margin + self.hparams.perturb_weight * g_loss_perturb
            
            # Track mean adversarial logit for monitoring
            with torch.no_grad():
                adv_logit_mean = adv_logits_g.mean()
                margin_violation_rate = (adv_logits_g > margin_tensor).float().mean()
                margin_violation_mean = torch.clamp(adv_logits_g - margin_tensor, min=0).mean()
        
        # Step 2: Update Generator (only if we have real images to generate adversarial samples)
        # Must include backward AND step together for AMP scaler compatibility
        if num_real > 0:
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_gradients(g_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            g_opt.step()
        
        # Log metrics
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/d_acc_real', d_acc_real, on_step=False, on_epoch=True)
        self.log('train/d_acc_fake', d_acc_fake, on_step=False, on_epoch=True)
        self.log('train/adv_logit_mean', adv_logit_mean, on_step=False, on_epoch=True)  # Should plateau, not collapse
        self.log('train/loss_real', loss_real, on_step=False, on_epoch=True)
        self.log('train/loss_fake', loss_fake, on_step=False, on_epoch=True)
        self.log('train/loss_consistency', loss_consistency, on_step=True, on_epoch=True)
        self.log('train/margin_violation_rate', margin_violation_rate, on_step=False, on_epoch=True)
        self.log('train/margin_violation_mean', margin_violation_mean, on_step=False, on_epoch=True)
        self.log('train/perturb_energy', perturb_energy, on_step=False, on_epoch=True)
        
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
        # Discriminator optimizer (lower LR - defensive stance)
        d_lr = self.hparams.lr * self.hparams.d_lr_mult
        d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )
        
        # Generator optimizer (higher LR - let it lead)
        g_lr = self.hparams.lr * self.hparams.g_lr_mult
        g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=g_lr,
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
    print("=" * 60)
    print("Testing DeepfakeGAN module...")
    print("=" * 60)
    
    # Create dummy batch
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nBatch size: {batch_size}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: Real={torch.sum(labels == 1).item()}, Fake={torch.sum(labels == 0).item()}")
    
    # ===== Test with Dual-Stream Discriminator =====
    print("\n--- Testing with Dual-Stream Discriminator ---")
    model_dual = DeepfakeGAN(
        d_backbone='convnext_tiny',
        d_pretrained=False,
        d_type='dual_stream',
        d_fusion_type='concat',
        g_base_channels=64,
        epsilon=0.03
    )
    
    # Quick forward test
    with torch.no_grad():
        logits = model_dual(images)
    print(f"Dual-stream output shape: {logits.shape}")
    print(f"Discriminator type: {type(model_dual.discriminator).__name__}")
    
    # ===== Test with Single-Stream Discriminator =====
    print("\n--- Testing with Single-Stream Discriminator (Legacy) ---")
    model_single = DeepfakeGAN(
        d_backbone='convnext_tiny',
        d_pretrained=False,
        d_type='single_stream',
        g_base_channels=64,
        epsilon=0.03
    )
    
    with torch.no_grad():
        logits = model_single(images)
    print(f"Single-stream output shape: {logits.shape}")
    print(f"Discriminator type: {type(model_single.discriminator).__name__}")

    print("\n" + "=" * 60)
    print("✓ DeepfakeGAN module test passed!")
    print("=" * 60)
