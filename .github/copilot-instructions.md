Below is a **brutally explicit `instructions.md`** meant for **GitHub Copilot / human contributors**.
It assumes the reader already understands PyTorch + Lightning and tells them **exactly what to change, why, and what *not* to touch**.

This document is written to **fix the conceptual flaw** in your current training while preserving **architecture, dataset, and intent**.

---

# DDGAN – Training Objective Refactor Instructions

## ⚠️ Context (Read First – Non-Negotiable)

This project **is not a GAN in the classical sense**.

* The generator **does not synthesize fake images**
* It generates **bounded adversarial perturbations**
* The discriminator **must remain a strong classifier**
* The goal is **adversarial robustness**, not GAN equilibrium

The current implementation incorrectly uses **Binary Cross Entropy on adversarial samples**, which:

* Compresses discriminator logits
* Destroys ranking ability
* Causes ROC-AUC collapse
* Leads to trivial equilibria

**This document defines the required changes to fix that.**

---

## 1. High-Level Goal of the Refactor

### Replace this (current behavior):

> “Train D to classify adversarial images as REAL using BCE”

### With this (correct behavior):

> “Train D to produce **consistent predictions** between clean and adversarial images, while preserving class separability”

This is **standard adversarial robustness training**, not GAN training.

---

## 2. What Must NOT Change

Do **not** modify the following unless explicitly instructed:

* DCT feature extractor
* ConvNeXt discriminator architecture
* Generator architecture (U-Net, frequency bottleneck, epsilon constraint)
* Dataset, transforms, or dataloaders
* Mixed precision, gradient clipping, or optimizer types
* Manual optimization structure in Lightning

If you change these, you are debugging the wrong thing.

---

## 3. Core Change: Discriminator Loss Redesign

### ❌ REMOVE (Current Code)

```python
# REMOVE THIS ENTIRE BLOCK
adv_outputs_d = self.discriminator(adv_images.detach())
adv_real_labels = torch.ones(num_real, 1, device=self.device)
loss_adv = self.criterion(adv_outputs_d, adv_real_labels)
```

And remove `loss_adv` from `d_loss`.

---

### ✅ ADD: Consistency Loss (Required)

#### Concept

The discriminator should output **similar probabilities** for:

* clean real image `x`
* adversarial real image `x_adv`

This enforces robustness **without destroying margins**.

---

### 🔧 Implementation

Add the following helper function:

```python
def consistency_loss(self, logits_clean, logits_adv):
    """
    Penalize prediction drift under adversarial perturbation.
    Uses probability space (after sigmoid).
    """
    p_clean = torch.sigmoid(logits_clean)
    p_adv = torch.sigmoid(logits_adv)
    return torch.mean((p_clean - p_adv) ** 2)
```

---

### 🔁 Modify Discriminator Training Step

Replace discriminator training logic with:

```python
# --- Discriminator forward ---
real_logits = self.discriminator(real_images)
fake_logits = self.discriminator(fake_images)

# Standard classification losses
loss_real = self.criterion(real_logits, torch.ones_like(real_logits))
loss_fake = self.criterion(fake_logits, torch.zeros_like(fake_logits))

# Generate adversarial images
adv_images, _ = self.generator(real_images)

# Forward adversarial images (NO LABELS)
adv_logits = self.discriminator(adv_images.detach())

# Consistency loss (NEW)
loss_consistency = self.consistency_loss(real_logits.detach(), adv_logits)

# Final discriminator loss
d_loss = loss_real + loss_fake + self.hparams.consistency_weight * loss_consistency
```

---

### 📌 Required New Hyperparameter

Add to config / hparams:

```yaml
consistency_weight: 1.0
```

**Do not exceed 2.0 initially** — higher values will again collapse logits.

---

## 4. Generator Objective Fix (Critical)

### ❌ REMOVE (Current Generator Adversarial Loss)

```python
adv_fake_labels = torch.zeros(...)
g_loss_adv = self.criterion(adv_outputs_g, adv_fake_labels)
```

This is label-based fooling and is **incorrect for robustness training**.

---

### ✅ ADD: Margin-Based Generator Loss

#### Concept

The generator should:

* Reduce discriminator confidence
* Without requiring full class flip
* Stay within epsilon bounds

---

### 🔧 Implementation

```python
def generator_margin_loss(self, logits, labels, margin=1.0):
    """
    Reduce discriminator confidence margin.
    labels: 1 for real, 0 for fake
    """
    signed_logits = (2 * labels - 1) * logits
    return torch.mean(torch.relu(margin - signed_logits))
```

---

### 🔁 Generator Training Step (Updated)

```python
adv_images, perturbation = self.generator(real_images)
adv_logits = self.discriminator(adv_images)

# Margin loss instead of BCE
labels_real = torch.ones_like(adv_logits)
g_loss_margin = self.generator_margin_loss(adv_logits, labels_real)

# Perturbation regularization (unchanged)
g_loss_perturb = torch.mean(torch.abs(perturbation))

# Final generator loss
g_loss = g_loss_margin + self.hparams.perturb_weight * g_loss_perturb
```

---

## 5. Validation: Fix the Metric Mismatch

### ❌ Current Problem

Validation only evaluates:

* Clean accuracy
* Clean AUC

This **cannot reflect robustness**.

---

### ✅ Required Addition: Adversarial Validation

In `validation_step`:

1. Generate adversarial images using **frozen generator**
2. Compute metrics for:

   * Clean images
   * Adversarial images

---

### 🔧 Minimal Required Change

```python
with torch.no_grad():
    adv_images, _ = self.generator(images)
    adv_logits = self.discriminator(adv_images)
```

Store:

* `clean_probs`
* `adv_probs`

---

### 📊 Required New Metrics

Log **both**:

* `val/auc_clean`
* `val/auc_adv`
* `val/robustness_gap = auc_clean - auc_adv`

If robustness gap shrinks over epochs → training works.

---

## 6. Logging Changes (Mandatory)

### Add These Logs

```python
self.log("train/loss_consistency", loss_consistency)
self.log("train/g_loss_margin", g_loss_margin)
self.log("val/auc_adv", auc_adv)
self.log("val/robustness_gap", auc_clean - auc_adv)
```

---

## 7. Expected Training Behavior (Reality Check)

After refactor:

* Accuracy may **drop slightly**
* AUC will:

  * Initially drop
  * Then stabilize
* Generator loss will **not go to zero**
* Discriminator logits will retain variance
* Adversarial validation will improve over epochs

If you expect GAN-style oscillations, you misunderstood the task.

---

## 8. Final Warning (Read This Twice)

> **DO NOT** reintroduce BCE on adversarial labels
> **DO NOT** force adversarial images to be “real” or “fake”
> **DO NOT** judge success using accuracy alone

If you ignore this, you will recreate the same failure mode.

---

## 9. What This Refactor Achieves

* Fixes theoretical inconsistency
* Preserves discriminator separability
* Enables meaningful ROC-AUC
* Makes robustness measurable
* Aligns implementation with literature (TRADES-style training)

Alright. I’m not going to ask you to choose — I’ll do the **most valuable next step**.

Below is a **concrete patch-style implementation guide**:
what files to touch, what functions to add, and **exact code replacements**.
If you follow this literally, you will fix the core failure mode.

This is written so **Copilot can autocomplete safely** and **you don’t accidentally reintroduce the bug**.

---

# PATCH GUIDE demonstrate

## Scope of this patch

**Goal:**
Convert your current *label-based adversarial training* into **proper robustness training** while keeping:

* DCT + ConvNeXt discriminator
* U-Net generator
* Lightning manual optimization
* Dataset and logging infra

**Files touched (typical):**

* `models/ddgan_module.py` (or equivalent LightningModule)
* `configs/*.yaml` (hyperparameters only)

---

## 1. Add new loss utilities (top of LightningModule)

### 🔧 ADD (do not modify existing imports)

```python
import torch.nn.functional as F
```

### 🔧 ADD BELOW your existing helper methods

```python
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
```

---

## 2. Fix DISCRIMINATOR training (critical)

### ❌ DELETE this block entirely

```python
# Adversarial robustness (WRONG)
adv_outputs_d = self.discriminator(adv_images.detach())
adv_real_labels = torch.ones(num_real, 1, device=self.device)
loss_adv = self.criterion(adv_outputs_d, adv_real_labels)
```

And **remove `loss_adv` from `d_loss`**.

---

### ✅ REPLACE discriminator section with this

```python
# ========== DISCRIMINATOR ==========
real_logits = self.discriminator(real_images)
fake_logits = self.discriminator(fake_images)

loss_real = self.criterion(real_logits, torch.ones_like(real_logits))
loss_fake = self.criterion(fake_logits, torch.zeros_like(fake_logits))

# Generate adversarial images
adv_images, _ = self.generator(real_images)
adv_logits = self.discriminator(adv_images.detach())

# Consistency loss (NEW)
loss_consistency = self.consistency_loss(real_logits.detach(), adv_logits)

# Final discriminator loss
d_loss = (
    loss_real
    + loss_fake
    + self.hparams.consistency_weight * loss_consistency
)
```

---

### 📌 Log it (mandatory)

```python
self.log("train/loss_consistency", loss_consistency, on_step=True, on_epoch=True)
```

---

## 3. Fix GENERATOR training (second critical fix)

### ❌ DELETE this

```python
adv_fake_labels = torch.zeros(num_real, 1, device=self.device)
g_loss_adv = self.criterion(adv_outputs_g, adv_fake_labels)
```

---

### ✅ REPLACE generator step with this

```python
# ========== GENERATOR ==========
adv_images_g, perturbation = self.generator(real_images)
adv_logits_g = self.discriminator(adv_images_g)

labels_real = torch.ones_like(adv_logits_g)

g_loss_margin = self.generator_margin_loss(
    adv_logits_g,
    labels_real,
    margin=self.hparams.margin
)

g_loss_perturb = torch.mean(torch.abs(perturbation))

g_loss = (
    g_loss_margin
    + self.hparams.perturb_weight * g_loss_perturb
)
```

### 📌 Log it

```python
self.log("train/g_loss_margin", g_loss_margin, on_step=True, on_epoch=True)
```

---

## 4. Validation MUST include adversarial evaluation

Right now your validation is **incomplete**.

### ❌ Current problem

You only validate on **clean images**, which tells you nothing about robustness.

---

### ✅ MODIFY `validation_step`

```python
# Clean
logits_clean = self.discriminator(images)
probs_clean = torch.sigmoid(logits_clean)

# Adversarial (NO gradients)
with torch.no_grad():
    adv_images, _ = self.generator(images)
    logits_adv = self.discriminator(adv_images)
    probs_adv = torch.sigmoid(logits_adv)

self.validation_step_outputs.append({
    "labels": labels.cpu(),
    "probs_clean": probs_clean.cpu().squeeze(),
    "probs_adv": probs_adv.cpu().squeeze(),
})
```

---

### ✅ MODIFY `on_validation_epoch_end`

Add:

```python
auc_clean = roc_auc_score(all_labels, all_probs_clean)
auc_adv = roc_auc_score(all_labels, all_probs_adv)
robust_gap = auc_clean - auc_adv
```

Log:

```python
self.log("val/auc_clean", auc_clean, prog_bar=True)
self.log("val/auc_adv", auc_adv, prog_bar=True)
self.log("val/robustness_gap", robust_gap, prog_bar=True)
```

---

## 5. Config changes (minimal, REQUIRED)

### 🔧 ADD to your config / hparams

```yaml
consistency_weight: 1.0
margin: 1.0
```

### ⚠️ Do NOT tune these yet

If you touch these before confirming behavior, you’ll mask bugs.

---

## 6. Expected behavior after patch (read this carefully)

If this is working:

* **Accuracy may drop** → expected
* **AUC_clean stays stable**
* **AUC_adv improves steadily**
* **Robustness gap shrinks**
* Generator loss **does not converge to zero**
* No flatlining of discriminator logits

If you see:

* AUC collapsing again → you reintroduced BCE-on-adv
* Generator loss → 0 → perturbation penalty too strong
* Both AUCs identical → generator ineffective

---

## 7. Final blunt warning

If you **skip adversarial validation**, this entire project is unprovable.

If you **reintroduce labels for adversarial images**, the theory collapses.

If you **optimize accuracy**, you will destroy robustness.

---
