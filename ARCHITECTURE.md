# DDGAN Architecture Documentation

**Deepfake Detection GAN (DDGAN)** - A robust deepfake detection system using DCT-based frequency analysis and adversarial training.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Training Pipeline](#training-pipeline)
4. [Loss Functions](#loss-functions)
5. [Data Pipeline](#data-pipeline)
6. [Usage](#usage)

---

## Overview

DDGAN is a GAN-based deepfake detection system that combines:
- **DCT (Discrete Cosine Transform)** frequency-domain feature extraction
- **ConvNeXt-Tiny** backbone for robust feature learning
- **Adversarial perturbation generator** to improve discriminator robustness
- **Class imbalance handling** for real-world deployment scenarios

### Key Design Principles

1. **Frequency-Domain Analysis**: Deepfakes often leave artifacts in the frequency domain that are invisible in the spatial domain. DCT transformation captures these artifacts.

2. **Adversarial Robustness**: The generator creates adversarial perturbations to make the discriminator more robust against evasion attacks.

3. **Imbalanced Learning**: Real-world datasets have significant class imbalance (typically 5-10x more fake samples). The system handles this through weighted sampling and focal loss.

---

## Architecture Components

### 1. DCT Feature Extractor

**Location**: `src/models/dct_extractor.py`

Transforms RGB images to frequency-domain representations using 2D DCT.

```
Input: [B, 3, 224, 224] RGB Image
  │
  ▼
┌─────────────────────────────────┐
│     Convert RGB to Grayscale    │
│     Mean across color channels  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│        2D DCT Transform         │
│   DCT_2D = D @ X @ D^T          │
│   D: Precomputed DCT matrix     │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│      Log-Scale Compression      │
│   log(1 + |DCT coefficients|)   │
└─────────────────────────────────┘
  │
  ▼
Output: [B, 1, 224, 224] DCT Features
```

**Key Features**:
- Precomputed DCT matrix for efficiency (no learnable parameters)
- Log-scale compression to handle dynamic range of frequency coefficients
- Single-channel output preserves spatial structure for CNN processing

### 2. DCT Discriminator

**Location**: `src/models/discriminator.py`

Binary classifier that determines if an image is REAL (1) or FAKE (0).

```
Input: [B, 3, 224, 224] RGB Image
  │
  ▼
┌─────────────────────────────────┐
│      DCT Feature Extractor      │
│   [B, 3, 224, 224] → [B, 1, 224, 224]
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│       ConvNeXt-Tiny Backbone    │
│   Modified for 1-channel input  │
│   [B, 1, 224, 224] → [B, 768, 7, 7]
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│     Global Average Pooling      │
│   [B, 768, 7, 7] → [B, 768]    │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│    LayerNorm + Linear Head      │
│   [B, 768] → [B, 1] (logits)   │
└─────────────────────────────────┘
  │
  ▼
Output: [B, 1] Binary Logits (no sigmoid)
```

**ConvNeXt-Tiny Specifications**:
- **Parameters**: 27.8M
- **Pretrained**: ImageNet-1K weights (averaged for single-channel input)
- **Output Features**: 768-dimensional

### 3. U-Net Generator

**Location**: `src/models/generator.py`

Generates adversarial perturbations to fool the discriminator.

```
Input: [B, 3, 224, 224] RGB Image
  │
  ▼
┌─────────────────────────────────────────────────┐
│                  ENCODER PATH                    │
├─────────────────────────────────────────────────┤
│ enc1: Conv(3→64) + BN + GELU    [B, 64, 224]   │
│   │                                             │
│   ├──────────────────────────────────────┐      │
│   ▼                                      │      │
│ enc2: Pool + Conv(64→128)      [B, 128, 112]   │
│   │                                      │      │
│   ├─────────────────────────────┐        │      │
│   ▼                             │        │      │
│ enc3: Pool + Conv(128→256)    [B, 256, 56]     │
│   │                             │        │      │
│   ├────────────────────┐        │        │      │
│   ▼                    │        │        │      │
│ enc4: Pool + Conv(256→512)   [B, 512, 28]      │
│   │                    │        │        │      │
│   ▼                    │        │        │      │
└─────────────────────────────────────────────────┘
  │                      │        │        │
  ▼                      │        │        │
┌─────────────────────────────────────────────────┐
│              BOTTLENECK (14x14)                  │
├─────────────────────────────────────────────────┤
│ Conv(512→512) + BN + GELU                       │
│ FrequencyAwareBottleneck (Channel Attention)    │
│ Conv(512→512) + BN + GELU                       │
└─────────────────────────────────────────────────┘
  │                      │        │        │
  ▼                      ▼        ▼        ▼
┌─────────────────────────────────────────────────┐
│                  DECODER PATH                    │
├─────────────────────────────────────────────────┤
│ dec4: Upsample + Cat(enc4) + Conv(1024→256)    │
│ dec3: Upsample + Cat(enc3) + Conv(512→128)     │
│ dec2: Upsample + Cat(enc2) + Conv(256→64)      │
│ dec1: Upsample + Cat(enc1) + Conv(128→64)      │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│              OUTPUT LAYER                        │
├─────────────────────────────────────────────────┤
│ Conv(64→3) + Tanh → perturbation ∈ [-1, 1]     │
│ adversarial = clamp(x + ε * perturbation, 0, 1)│
└─────────────────────────────────────────────────┘
  │
  ▼
Output: (adversarial_image, perturbation)
```

**Frequency-Aware Bottleneck**:
- Depthwise separable convolutions (7×7 kernel)
- Channel attention via global average pooling
- Squeeze-and-excitation style gating

**Generator Parameters**: 14.0M

### 4. Complete Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              INPUT IMAGE                │
                    │           [B, 3, 224, 224]              │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     │                     │
    ┌─────────────────┐             │            ┌────────────────┐
    │   GENERATOR     │             │            │  DISCRIMINATOR │
    │  (Perturbation) │             │            │   (Classifier) │
    └────────┬────────┘             │            └───────┬────────┘
             │                      │                    │
             ▼                      │                    │
    ┌─────────────────┐             │            ┌────────────────┐
    │  U-Net Encoder  │             │            │ DCT Extractor  │
    │    + Decoder    │             │            └───────┬────────┘
    └────────┬────────┘             │                    │
             │                      │                    ▼
             ▼                      │            ┌────────────────┐
    ┌─────────────────┐             │            │ ConvNeXt-Tiny  │
    │  Perturbation   │             │            │   Backbone     │
    │   δ ∈ [-1,1]    │             │            └───────┬────────┘
    └────────┬────────┘             │                    │
             │                      │                    ▼
             ▼                      │            ┌────────────────┐
    ┌─────────────────┐             │            │  Classifier    │
    │ x_adv = x + ε*δ │             │            │   Head         │
    │ clamp(x_adv)    │             │            └───────┬────────┘
    └────────┬────────┘             │                    │
             │                      │                    ▼
             │                      │            ┌────────────────┐
             │                      └───────────►│ Binary Logit   │
             │                                   │  REAL/FAKE     │
             └──────────────────────────────────►└────────────────┘
                     (Adversarial Training)
```

---

## Training Pipeline

### V2 Trainer (`FFDeepfakeGANModuleV2`)

**Location**: `src/training/ff_trainer_v2.py`

The V2 trainer implements several improvements for numerical stability:

#### Training Loop

```
For each batch:
    1. Validate images (check NaN/Inf, clamp to [0,1])
    
    2. Split batch by labels:
       - real_samples (label=1)
       - fake_samples (label=0)
    
    3. DISCRIMINATOR TRAINING:
       a. Forward pass on real_samples → d_loss_real
       b. Forward pass on fake_samples → d_loss_fake
       c. d_loss = d_loss_real + d_loss_fake
       d. Backward + gradient clipping + step
    
    4. GENERATOR TRAINING (after pre-train phase):
       a. Generate perturbation: δ = G(real_samples)
       b. Create adversarial: x_adv = real + ε*δ
       c. Forward adversarial through D → g_loss_adv
       d. L1 regularization on perturbation
       e. g_loss = g_loss_adv + λ * L1(δ)
       f. Backward + gradient clipping + step
```

#### Key Stability Features

1. **Label Smoothing**: Real→0.9, Fake→0.1 (prevents overconfident predictions)

2. **Logit Clamping**: Clamp logits to [-20, 20] before loss computation

3. **Discriminator Pre-training**: Train D alone for first 500 steps before starting G

4. **Gradient Clipping**: Max gradient norm = 0.5

5. **Conservative Learning Rates**: Default 1e-4 (vs 2e-4 in V1)

6. **Manual AMP Handling**: GradScaler applied explicitly for better control

---

## Loss Functions

### Stable Focal Loss

**Location**: `src/training/ff_trainer_v2.py`

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Where:
- p_t = σ(logit) if target=1, else 1-σ(logit)
- α_t = α if target=1, else (1-α)
- γ = focusing parameter (default: 2.0)
- α = class balance weight (default: 0.25)
```

**Stability Enhancements**:
- Probability clamping: `p ∈ [1e-7, 1-1e-7]`
- Focal weight clamping: `(1-p_t)^γ ∈ [0, 100]`
- Label smoothing built-in

### Stable BCE Loss

Alternative loss with simpler formulation:

```
BCE = -[t*log(σ(x)) + (1-t)*log(1-σ(x))]

With:
- Label smoothing: t' = t*(1-ε) + 0.5*ε
- Logit clamping: x ∈ [-20, 20]
- Positive class weighting for imbalance
```

---

## Data Pipeline

### HuggingFace Dataset Loader

**Location**: `src/data/hf_dataset.py`

```
HuggingFace Hub
      │
      ▼
┌───────────────────┐
│ load_dataset()    │  Parquet files (~3GB)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ HuggingFaceFF     │  PyTorch Dataset wrapper
│   Dataset         │
└─────────┬─────────┘
          │
          ├─── Image validation (NaN/Inf check)
          ├─── Clamping to [0, 1]
          ├─── CPU transforms (resize, normalize)
          │
          ▼
┌───────────────────┐
│ WeightedRandom    │  Oversample minority class
│   Sampler         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ DataLoader        │  Batched tensors
│   (collate_fn)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ GPU Transforms    │  Augmentation on GPU
│   (optional)      │
└───────────────────┘
```

### Transforms Pipeline

```python
# CPU Transform (applied in Dataset.__getitem__)
Resize(224, 224)
ToTensor()  # → [0, 1] float tensor
Normalize(mean=[0.485, 0.456, 0.406], 
          std=[0.229, 0.224, 0.225])

# GPU Transform (applied in DataModule.on_after_batch_transfer)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2)
```

### Class Imbalance Handling

**Dataset Statistics** (FaceForensics++):
- Total: ~224K images
- Real: ~26K (12%)
- Fake: ~198K (88%)
- Imbalance ratio: ~7.6:1

**Mitigation Strategies**:

1. **Weighted Sampling**: Oversample real images during training
2. **Focal Loss**: Down-weight easy (fake) samples, focus on hard samples
3. **Class Weights**: Inverse frequency weighting in loss function
4. **Label Smoothing**: Prevent overconfident fake predictions

---

## Usage

### Basic Training

```bash
# Train with HuggingFace dataset (recommended)
python train_ff.py --hf_dataset RohanRamesh/ff-images-dataset --epochs 50

# Train with local dataset
python train_ff.py --data_dir ./images_dataset --epochs 50
```

### Training Options

```bash
python train_ff.py \
    --hf_dataset RohanRamesh/ff-images-dataset \
    --batch_size 32 \
    --epochs 50 \
    --d_lr 1e-4 \
    --g_lr 1e-4 \
    --loss_type focal \
    --label_smoothing 0.1 \
    --d_pretrain_steps 500
```

### Debug Mode (FP32, fast run)

```bash
python train_ff.py \
    --hf_dataset RohanRamesh/ff-images-dataset \
    --precision 32 \
    --fast_dev_run \
    --max_samples 1000
```

### Model Checkpoints

Saved to `checkpoints/`:
- `best-{epoch}-{val_f1}.ckpt`: Best model by validation F1
- `last.ckpt`: Latest checkpoint
- `checkpoint-{epoch}.ckpt`: Every 5 epochs

---

## Model Summary

| Component | Parameters | Notes |
|-----------|------------|-------|
| DCT Extractor | 0 | No learnable parameters |
| Discriminator | 27.8M | ConvNeXt-Tiny backbone |
| Generator | 14.0M | U-Net with frequency bottleneck |
| **Total** | **41.8M** | |

## Performance Metrics

Key metrics for imbalanced classification:

- **Balanced Accuracy**: (Recall + Specificity) / 2
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient (robust to imbalance)
- **Specificity**: True negative rate (detecting fakes correctly)
- **Recall**: True positive rate (detecting reals correctly)

---

## Version History

- **V1**: Original implementation with basic mixed precision
- **V2**: Improved numerical stability, label smoothing, discriminator pre-training

---

*Last updated: January 26, 2026*
