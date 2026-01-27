Deepfake Detection GAN - Complete Workflow Documentation
🎯 Project Goal
Build a robust deepfake detector using a GAN-based adversarial training framework where:

The discriminator learns to detect deepfake images using frequency-domain (DCT) features
The generator creates adversarial perturbations to make the discriminator more robust
The system becomes resilient to subtle manipulations through adversarial training


📊 Dataset
Source: celebDF v2 images
Specifications:

Images: 224×224 pixels, RGB

Data Characteristics:

Contains both real images and AI-generated fakes

Pre-sized to 224×224 (no resizing needed during training)


🏗️ Architecture Design
Design Philosophy
Why DCT instead of raw pixels?

Deepfake artifacts are more visible in frequency domain
JPEG compression patterns differ between real and generated images
DCT captures high-frequency anomalies that GANs struggle to replicate
More compact representation of manipulation signatures

Why grayscale DCT?

Simplifies the feature space (1 channel vs 3)
Luminance contains most structural information
Faster computation than per-channel DCT
Reduces model complexity

Why adversarial training?

Creates a robust detector that handles edge cases
Generator learns to find discriminator weaknesses
Results in better generalization to unseen deepfakes
Mimics real-world adversarial attacks


1. DCT Feature Extractor
Mathematical Foundation:
DCT-II Transform Matrix:
dct_matrix[k,n] = cos(π × k × (2n + 1) / (2 × N))

Normalization:
- First row: sqrt(1/N)
- Other rows: sqrt(2/N)
Implementation:
pythonclass DCT2D(nn.Module):
    - Precomputes DCT basis matrix (224×224)
    - Stored as buffer (non-trainable, moves to GPU automatically)
    - Applies separable 2D DCT:
      1. Matrix multiply along rows
      2. Matrix multiply along columns
    - Complexity: O(N² log N) vs O(N⁴) for naive implementation
```

**Feature Extraction Pipeline**:
```
RGB Image [B, 3, 224, 224]
    ↓
Grayscale conversion: 0.299R + 0.587G + 0.114B
    ↓
Grayscale [B, 1, 224, 224]
    ↓
2D DCT transform
    ↓
DCT coefficients [B, 1, 224, 224]
    ↓
Log scaling: log(|DCT| + 1e-8)
    ↓
Output [B, 1, 224, 224]
```

**Why log scaling?**
- DCT coefficients have huge dynamic range (DC component >> AC components)
- Log compression makes features more learnable
- Similar to how human perception is logarithmic
- Prevents gradient domination by DC component

---

### **2. Discriminator Architecture**

**High-level Flow**:
```
RGB Image [B, 3, 224, 224]
    ↓
DCT Feature Extractor → [B, 1, 224, 224]
    ↓
ConvNeXt-Tiny Backbone
    ├─ Stem: Conv 4×4, stride 4 → [B, 96, 56, 56]
    ├─ Stage 1: 3 blocks → [B, 96, 56, 56]
    ├─ Stage 2: 3 blocks → [B, 192, 28, 28]
    ├─ Stage 3: 9 blocks → [B, 384, 14, 14]
    └─ Stage 4: 3 blocks → [B, 768, 7, 7]
    ↓
Global Average Pooling → [B, 768]
    ↓
LayerNorm + Linear(768 → 1)
    ↓
Logits [B, 1] (real/fake probability)
```

**Key Design Decisions**:

**ConvNeXt-Tiny**:
- Modern CNN architecture (2022)
- 27.8M parameters
- Better than ResNet for vision tasks
- Efficient inference
- Pretrained on ImageNet

**Input adaptation**:
- Original ConvNeXt: 3-channel RGB input
- Modified: 1-channel DCT input
- Initialization: Average RGB weights across channels
- Preserves pretrained knowledge for grayscale features

**Why pretrained weights?**
- Transfer learning from ImageNet
- Learns low-level features faster
- Better generalization
- Reduces training time

**Classification head**:
- Global average pooling: Reduces spatial dimensions
- LayerNorm: Stabilizes training
- Single linear layer: Binary classification
- No sigmoid: Using BCEWithLogitsLoss (numerically stable)

---

### **3. Generator Architecture**

**Design**: U-Net with Frequency-Aware Bottleneck

**Why U-Net?**
- Skip connections preserve spatial details
- Encoder-decoder structure for image-to-image translation
- Widely used in image manipulation tasks
- Effective for generating perturbations

**Architecture**:
```
Input: RGB Image [B, 3, 224, 224]
    ↓
ENCODER (Downsampling Path)
    ├─ enc1: Conv blocks → [B, 64, 224, 224]
    ├─ pool → [B, 64, 112, 112]
    ├─ enc2: Conv blocks → [B, 128, 112, 112]
    ├─ pool → [B, 128, 56, 56]
    ├─ enc3: Conv blocks → [B, 256, 56, 56]
    ├─ pool → [B, 256, 28, 28]
    ├─ enc4: Conv blocks → [B, 512, 28, 28]
    └─ pool → [B, 512, 14, 14]
    ↓
BOTTLENECK (Frequency-Aware Processing)
    ├─ Conv2d(512 → 512)
    ├─ FrequencyAwareBottleneck
    └─ Conv2d(512 → 512)
    → [B, 512, 14, 14]
    ↓
DECODER (Upsampling Path with Skip Connections)
    ├─ upsample + concat(e4) → [B, 1024, 28, 28]
    ├─ dec4 → [B, 256, 28, 28]
    ├─ upsample + concat(e3) → [B, 512, 56, 56]
    ├─ dec3 → [B, 128, 56, 56]
    ├─ upsample + concat(e2) → [B, 256, 112, 112]
    ├─ dec2 → [B, 64, 112, 112]
    ├─ upsample + concat(e1) → [B, 128, 224, 224]
    └─ dec1 → [B, 64, 224, 224]
    ↓
OUTPUT LAYER
    Conv2d + Tanh → perturbation [B, 3, 224, 224] ∈ [-1, 1]
    ↓
ADVERSARIAL IMAGE GENERATION
    adversarial = input + ε × perturbation
    adversarial = clamp(adversarial, 0, 1)
    ↓
Output: (adversarial_image, perturbation)
Encoder/Decoder Blocks:
pythonEach block:
    Conv2d(3×3, padding=1)
    BatchNorm2d
    GELU activation
    Conv2d(3×3, padding=1)
    BatchNorm2d
    GELU activation
```

**FrequencyAwareBottleneck**:
```
Input [B, 512, H, W]
    ↓
Depthwise Conv (7×7, groups=512)
    ↓
Pointwise Conv (1×1)
    ↓
BatchNorm + GELU
    ↓
Frequency Gate:
    ├─ Global Average Pooling → [B, 512, 1, 1]
    ├─ Conv (512 → 32)
    ├─ GELU
    ├─ Conv (32 → 512)
    └─ Sigmoid → attention weights
    ↓
Apply gate: features × attention
    ↓
Second conv block (same structure)
    ↓
Residual connection: output = processed + input
Why this bottleneck design?

Large kernels (7×7): Capture frequency-like patterns without explicit FFT
Depthwise separable: Efficient computation, reduces parameters
Frequency gate: Learns which frequency bands to emphasize
No explicit FFT/IFFT: Avoids compilation issues, faster training
Residual connection: Enables gradient flow, learning identity + perturbation

Perturbation generation:

Tanh output: Bounds perturbation to [-1, 1]
Epsilon scaling: Controls maximum perturbation magnitude (default 0.03)
Clamping: Ensures final image stays in [0, 1] range
L1 regularization: Encourages sparse, small perturbations

🔄 Training Workflow
Training Philosophy
This is adversarial robustness training, not traditional GAN training:
Traditional GAN:

Generator: Create realistic fakes
Discriminator: Detect fakes
Goal: Generator fools discriminator completely

Our approach:

Generator: Create adversarial perturbations
Discriminator: Remain robust despite perturbations
Goal: Discriminator stays accurate even with adversarial attacks


Data Pipeline
CPU Side (Data Loading):
python1. Load from HuggingFace (Parquet format)
2. Convert PIL image to RGB
3. ToTensor + normalize to [0, 1]
4. No resizing (already 224×224)
5. Batch collation
GPU Side (During Training):
python1. Transfer batch to GPU (non_blocking=True)
2. Apply normalization: c stats
   mean=find mean of the dataset per channel
   std=find stddev of the dataset per channel
3. Extract DCT features (on-the-fly)
4. Forward pass
Why split CPU/GPU transforms?

Maximize CPU workers for I/O
GPU does compute-heavy operations (DCT, convolutions)
Overlapped data loading and training
Better hardware utilization

DataLoader settings:
pythonbatch_size = 32
num_workers = 4
pin_memory = True          # Fast CPU→GPU transfer
persistent_workers = True  # Keep workers alive between epochs
shuffle = True            # Random sampling

Training Loop (Single Step)
Input: Batch of mixed real and fake images with labels
Step 1: Separate real and fake samples
pythonreal_batch = images[labels == 1]  # Real images
fake_batch = images[labels == 0]  # Fake images
Step 2: Train Discriminator
pythonObjective: Classify correctly while being robust to adversarial perturbations

1. Forward real images
   real_outputs = D(real_images)
   loss_real = BCE(real_outputs, ones)
   
2. Forward fake images (detached from generator)
   fake_outputs = D(fake_images.detach())
   loss_fake = BCE(fake_outputs, zeros)
   
3. Generate adversarial images from real images
   adv_images, _ = G(real_images)
   adv_outputs = D(adv_images.detach())
   loss_adv = BCE(adv_outputs, ones)  # Should still classify as real
   
4. Combined discriminator loss
   d_loss = loss_real + loss_fake + 0.5 × loss_adv
   
5. Backward + optimizer step
Why 0.5 weight on adversarial loss?

Balances three objectives
Prevents adversarial samples from dominating
Discriminator should classify clean images well + be robust to perturbations

Step 3: Train Generator
pythonObjective: Create perturbations that fool discriminator while being minimal

1. Generate adversarial images
   adv_images, perturbation = G(real_images)
   
2. Forward through discriminator (no detach!)
   adv_outputs = D(adv_images)
   
3. Adversarial loss (fool discriminator)
   loss_adv = BCE(adv_outputs, zeros)  # Want D to classify as fake
   
4. Perturbation regularization (keep small)
   loss_perturb = mean(|perturbation|)  # L1 norm
   
5. Combined generator loss
   g_loss = loss_adv + 0.1 × loss_perturb
   
6. Backward + optimizer step
Why L1 regularization?

Encourages sparse perturbations
Prevents large, obvious modifications
Makes adversarial examples more subtle
L1 (vs L2) promotes exact zeros in perturbation

Step 4: Return metrics
python{
    'd_loss': discriminator loss value,
    'g_loss': generator loss value,
    'd_acc_real': accuracy on real images,
    'd_acc_fake': accuracy on fake images,
    'g_acc_adv': how often adversarial images fool discriminator
}

Optimization Details
Optimizers:
pythonAdamW for both models:
    lr = 2e-4              # Learning rate
    betas = (0.5, 0.999)   # Momentum parameters (β1 lower for GANs)
    weight_decay = 0.01    # L2 regularization
Why AdamW?

Decoupled weight decay (better than Adam)
Adaptive learning rates per parameter
Stable for GAN training
Industry standard for transformers/modern CNNs

Why β1=0.5?

Standard for GAN training
Reduces momentum to avoid oscillations
Helps with adversarial training stability

Learning rate scheduling:
pythonCosineAnnealingLR:
    - Gradually reduces learning rate
    - Follows cosine curve
    - Helps convergence in later epochs
    - T_max = num_epochs
Mixed Precision Training (AMP):
pythonAutomatic Mixed Precision (16-bit):
    - Uses float16 for forward/backward pass
    - Uses float32 for critical operations (loss, optimizer)
    - Reduces memory usage by ~50%
    - Speeds up training by ~2-3x on modern GPUs
    - Maintains numerical stability with loss scaling
Gradient management:
pythonGradient clipping (max_norm=1.0):
    - Prevents exploding gradients
    - Critical for GAN stability
    - Clips gradient norm to maximum value

Loss Functions
BCEWithLogitsLoss:
pythonWhy this over BCE?
    - Numerically stable (combines sigmoid + BCE)
    - Prevents log(0) errors
    - Better gradients
    - Standard for binary classification
    
Formula:
    loss = -[y × log(σ(x)) + (1-y) × log(1-σ(x))]
    where σ(x) = 1/(1 + e^(-x))
```

**Loss interpretation**:
```
Perfect prediction: loss ≈ 0
Random guess: loss ≈ 0.693 (log(2))
Completely wrong: loss → ∞

PyTorch Lightning Integration
Why Lightning?

Handles device management automatically
Built-in AMP support
Distributed training (multi-GPU)
Checkpointing and logging
Less boilerplate code
Professional best practices

Key components:
pythonLightningModule:
    - training_step(): Single training iteration
    - validation_step(): Single validation iteration
    - configure_optimizers(): Setup optimizers + schedulers
    - on_train_epoch_end(): Epoch-level logging
    
Trainer:
    - Orchestrates training loop
    - Handles precision, devices, callbacks
    - Automatic checkpointing
    - Progress bars, logging
Current configuration:
pythonTrainer(
    max_epochs = 50
    precision = "16-mixed"     # AMP
    accelerator = "gpu"
    devices = 2                # Dual GPU
    strategy = "ddp"           # Distributed Data Parallel
    val_check_interval = 1.0   # Validate every epoch
    log_every_n_steps = 50
)
```

---

## 📈 Training Dynamics

### **Expected Training Progression**

**Phase 1: Initial Learning (Epochs 0-5)**
```
Discriminator learns quickly:
    - d_acc_real: 0.5 → 0.85
    - d_acc_fake: 0.5 → 0.85
    - d_loss: 1.4 → 0.4
    
Generator struggles:
    - g_loss: high (1.5-2.5)
    - Perturbations ineffective
```

**Phase 2: Adversarial Competition (Epochs 5-20)**
```
Generator improves:
    - Learns effective perturbations
    - g_loss decreases
    - d_acc_real drops slightly (0.85 → 0.70)
    
Discriminator adapts:
    - Becomes robust to perturbations
    - Losses stabilize
```

**Phase 3: Equilibrium (Epochs 20-50)**
```
Both models reach Nash equilibrium:
    - d_acc: oscillates around 0.65-0.75
    - Losses relatively stable
    - Small improvements in validation accuracy
    - Discriminator generalizes better
Healthy vs Unhealthy Training
✅ Healthy signs:

Losses decrease initially then stabilize
Accuracies in 0.6-0.8 range
Gradual validation improvement
No wild oscillations

⚠️ Warning signs:

d_acc → 1.0: Discriminator too strong (generator collapse)
d_acc → 0.5: Discriminator too weak (not learning)
Loss = NaN: Gradient explosion (reduce LR, add clipping)
g_loss not decreasing: Generator stuck (increase g_lr)

Mode collapse:

Generator produces same perturbations
Discriminator accuracy doesn't improve
Solution: Restart with different initialization


🔍 Validation & Metrics
Validation loop:
pythonEvery epoch:
    1. Set models to eval mode
    2. Disable gradient computation
    3. Forward pass on validation set
    4. Compute metrics:
       - Accuracy
       - Precision (TP / (TP + FP))
       - Recall (TP / (TP + FN))
       - F1 Score (harmonic mean)
       - ROC-AUC
```

**Key metrics**:
```
Accuracy: Overall correctness
Precision: Of predicted fakes, how many are actually fake?
Recall: Of actual fakes, how many did we detect?
F1: Balance between precision and recall
ROC-AUC: Threshold-independent performance
```

**Target performance**:
```
Accuracy: > 0.80
Precision: > 0.75
Recall: > 0.75
F1: > 0.75
ROC-AUC: > 0.85

💾 Checkpointing Strategy
What's saved:
pythonCheckpoint contains:
    - discriminator.state_dict()
    - generator.state_dict()
    - d_optimizer.state_dict()
    - g_optimizer.state_dict()
    - scheduler states
    - epoch number
    - best validation metrics
```

**Checkpoint frequency**:
- Every 5 epochs (safety)
- Best validation accuracy (top-3 models)
- Final epoch

**File size**: ~170MB per checkpoint (41.8M params × 4 bytes/float32)

**Storage management**:
```
Keep:
    - 3 best checkpoints (by validation accuracy)
    - Last checkpoint
    - Every 10th epoch (milestones)
Delete:
    - Intermediate checkpoints
```

---

## ⚙️ Implementation Stack

**Core libraries**:
```
PyTorch 2.0+: Deep learning framework
torchvision: Image transforms, pretrained models
PyTorch Lightning: Training orchestration
```

**Data**:
```
HuggingFace datasets: Dataset loading and streaming
```

**Utilities**:
```
numpy: Numerical operations
matplotlib: Visualization
scikit-learn: Metrics computation
tqdm: Progress bars
```

**Hardware requirements**:
```
GPU: 2× with 16GB+ VRAM (Kaggle T4/P100)
RAM: 32GB+ (for dataset caching)
Storage: 50GB+ (dataset + checkpoints)

🎯 Current Status
What's working:
✅ Dataset loaded (537K train, 297K val)
✅ Models initialized (41.8M total params)
✅ Dual GPU detected
✅ AMP enabled (16-bit mixed precision)
✅ Lightning trainer configured
✅ Checkpointing enabled
Next steps:

⏳ Start training and monitor first 100 batches
⏳ Verify batch composition (real/fake mix)
⏳ Check DCT features are extracted correctly
⏳ Monitor for mode collapse or gradient issues
⏳ Adjust hyperparameters if needed

Expected completion:

Training time: ~10-15 min/epoch
50 epochs: ~8-12 hours total
Risk: Kaggle timeout at 9 hours
Mitigation: Reduce to 35 epochs or use checkpointing


🔬 Key Innovations in This Approach

DCT-based detection:

Most deepfake detectors use raw pixels or spatial features
DCT captures frequency artifacts invisible to spatial methods
More robust to post-processing (compression, resizing)


Adversarial robustness training:

Unlike supervised learning, this creates actively robust detectors
Generator finds weaknesses, discriminator learns to defend
Mimics real-world adversarial attacks


On-the-fly DCT computation:

No preprocessing required
Saves storage (no cached features)
Enables data augmentation in pixel space


Grayscale DCT simplification:

Reduces complexity vs 3-channel
Still captures essential frequency information
Faster training, lower memory


Frequency-aware generator:

Bottleneck learns frequency-domain patterns
No explicit FFT (compilation-friendly)
Implicit frequency filtering via large kernels



