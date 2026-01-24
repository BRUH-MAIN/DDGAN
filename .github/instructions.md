# GitHub Copilot Instructions for Deepfake Detection GAN Pipeline

## Project Overview
Build a GAN-based deepfake detector using DCT (Discrete Cosine Transform) features and adversarial robustness training. The discriminator uses grayscale DCT features fed into a ConvNeXt backbone. The generator creates adversarial perturbations to make the discriminator more robust.

---

## File Structure

```
deepfake-gan/
├── src/
│   ├── models/
│   │   ├── dct_extractor.py
│   │   ├── discriminator.py
│   │   └── generator.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── train.py
├── evaluate.py
├── config.py
└── requirements.txt
```

---

## 1. Create `src/models/dct_extractor.py`

**Instructions for Copilot:**

Create a PyTorch module called `DCT2D` that:
- Accepts a size parameter (default 224) in __init__
- Precomputes the DCT-II transformation matrix as a buffer (non-trainable)
- The DCT matrix formula: `dct_matrix[k,n] = cos(π * k * (2n + 1) / (2 * size))`
- First row is scaled by `sqrt(1/size)`, remaining rows by `sqrt(2/size)`
- Implements forward pass using separable 2D DCT via matrix multiplication
- Forward accepts input shape [B, C, H, W] and returns [B, C, H, W]
- Use torch.matmul for rows, then columns (separable transform)

Then create `DCTFeatureExtractor` module that:
- Uses DCT2D internally
- In forward pass:
  - Converts RGB input [B, 3, H, W] to grayscale using weights [0.299, 0.587, 0.114]
  - Adds channel dimension: [B, 1, H, W]
  - Applies DCT2D transform
  - Applies log scaling: `log(abs(dct) + 1e-8)`
  - Returns [B, 1, H, W]

---

## 2. Create `src/models/discriminator.py`

**Instructions for Copilot:**

Create a `DCTDiscriminator` class that:
- Inherits from nn.Module
- In __init__ (accepts pretrained=True parameter):
  - Instantiate DCTFeatureExtractor with size=224
  - Load ConvNeXt-Tiny using torchvision.models.convnext_tiny
  - If pretrained=True, use ConvNeXt_Tiny_Weights.DEFAULT
  - Modify the first conv layer to accept 1 input channel instead of 3
  - If pretrained, initialize the new conv by averaging the original RGB weights across the channel dimension
  - Store the backbone as self.backbone (convnext.features)
  - Store avgpool as self.avgpool (convnext.avgpool)
  - Create classifier as Sequential: LayerNorm(768), Linear(768, 1)

- In forward pass (input shape [B, 3, 224, 224]):
  - Extract DCT features using self.dct_extractor → [B, 1, 224, 224]
  - Pass through backbone → [B, 768, 7, 7]
  - Apply avgpool → [B, 768, 1, 1]
  - Flatten to [B, 768]
  - Pass through classifier → [B, 1]
  - Return logits (no sigmoid, we'll use BCEWithLogitsLoss)

Add a method `get_features` that returns the backbone features before classification for visualization.

---

## 3. Create `src/models/generator.py`

**Instructions for Copilot:**

Create a `FrequencyAwareBottleneck` module that:
- Accepts channels parameter (default 512)
- Has two sequential blocks (freq_conv1, freq_conv2), each containing:
  - Depthwise conv (kernel=7, padding=3, groups=channels)
  - Pointwise conv (kernel=1)
  - BatchNorm2d
  - GELU activation
- Has a frequency gate module (self.freq_gate):
  - AdaptiveAvgPool2d(1) for global context
  - Conv2d to reduce channels: channels → channels//16, kernel=1
  - GELU
  - Conv2d to expand: channels//16 → channels, kernel=1
  - Sigmoid activation
- Forward pass: process input through freq_conv1, apply gate, freq_conv2, add residual connection

Create a `UNetGenerator` class that:
- Accepts input_channels=3, output_channels=3
- Define helper method `_encoder_block(in_ch, out_ch)` returning Sequential:
  - Conv2d(in_ch, out_ch, kernel=3, padding=1)
  - BatchNorm2d
  - GELU
  - Conv2d(out_ch, out_ch, kernel=3, padding=1)
  - BatchNorm2d
  - GELU

- Define helper method `_decoder_block(in_ch, out_ch)` with same structure as encoder

- In __init__:
  - Create encoder blocks: enc1(3→64), enc2(64→128), enc3(128→256), enc4(256→512)
  - Create bottleneck as Sequential: Conv2d(512,512), BN, GELU, FrequencyAwareBottleneck(512), Conv2d(512,512), BN, GELU
  - Create decoder blocks: dec4(1024→256), dec3(512→128), dec2(256→64), dec1(128→64)
  - Create output layer: Conv2d(64, 3, kernel=3, padding=1), Tanh
  - Store MaxPool2d(2) as self.pool
  - Store Upsample(scale_factor=2, mode='bilinear', align_corners=True) as self.upsample

- Forward pass (accepts x [B,3,224,224] and epsilon=0.03):
  - Encode: e1 = enc1(x), e2 = enc2(pool(e1)), e3 = enc3(pool(e2)), e4 = enc4(pool(e3))
  - Bottleneck: b = bottleneck(pool(e4))
  - Decode with skip connections:
    - d4 = dec4(concat[upsample(b), e4])
    - d3 = dec3(concat[upsample(d4), e3])
    - d2 = dec2(concat[upsample(d3), e2])
    - d1 = dec1(concat[upsample(d2), e1])
  - Generate perturbation: perturbation = output(d1) → [B,3,224,224] in range [-1,1]
  - Create adversarial: adversarial = x + epsilon * perturbation
  - Clamp adversarial to [0, 1]
  - Return (adversarial, perturbation)

---

## 4. Create `src/data/dataset.py`

**Instructions for Copilot:**

Create a function `load_dataset(dataset_name, cache_dir=None)`:
- Use datasets.load_dataset to load from HuggingFace
- Return DatasetDict with 'train' and 'val' splits
- Handle cache_dir if provided

Create transforms in `src/data/transforms.py`:
- Define cpu_transform as torchvision Compose:
  - Lambda to convert PIL image to RGB
  - ToImage()
  - ToDtype(torch.float32, scale=True)

- Define a function `get_gpu_transform()` returning a nn.Module:
  - Resize to (224, 224) with antialias=True
  - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Create a function `preprocess_function(batch)`:
- Apply cpu_transform to each image in batch["image"]
- Return modified batch

Create a `collate_fn(batch)`:
- Stack images into tensor
- Extract labels as float32 tensor
- Return (images, labels)

---

## 5. Create `src/training/trainer.py`

**Instructions for Copilot:**

Create a `DeepfakeGANTrainer` class with:

**__init__ method** that accepts:
- discriminator: the discriminator model
- generator: the generator model
- device: torch device
- d_lr: discriminator learning rate (default 2e-4)
- g_lr: generator learning rate (default 2e-4)
- Setup:
  - Move models to device
  - Compile models with torch.compile()
  - Create AdamW optimizers for both (betas=(0.5, 0.999), weight_decay=0.01)
  - Create BCEWithLogitsLoss as self.criterion
  - Initialize schedulers: CosineAnnealingLR for both optimizers

**train_step method** that accepts (real_images, labels):
- Split batch into real (labels==1) and fake (labels==0) samples
- If either batch is empty, return None
- Train discriminator:
  - Zero grad
  - Forward real images, compute loss vs ones
  - Forward fake images (detached), compute loss vs zeros
  - Generate adversarial images from real using generator (detached)
  - Forward adversarial images, compute loss vs ones (should still detect as real)
  - Total d_loss = real_loss + fake_loss + 0.5 * adv_loss
  - Backward and step optimizer
- Train generator:
  - Zero grad
  - Generate adversarial images from real images
  - Forward through discriminator
  - Compute g_loss_adv: loss vs zeros (trying to fool discriminator)
  - Compute g_loss_perturb: mean absolute value of perturbation (L1 regularization)
  - Total g_loss = g_loss_adv + 0.1 * g_loss_perturb
  - Backward and step optimizer
- Return dictionary with: d_loss, g_loss, d_acc_real, d_acc_fake, g_acc_adv

**validate method** that accepts (val_loader):
- Set models to eval mode
- Use torch.no_grad()
- Iterate through val_loader
- Compute discriminator predictions on all images
- Calculate accuracy, precision, recall, F1 for binary classification
- Return metrics dictionary

**save_checkpoint method** that accepts (path, epoch, metrics):
- Save discriminator state_dict
- Save generator state_dict
- Save optimizer state_dicts
- Save epoch and metrics

**load_checkpoint method** that accepts (path):
- Load and restore all saved states

---

## 6. Create `src/training/losses.py`

**Instructions for Copilot:**

Create a `PerceptualLoss` class:
- Uses VGG16 features (layers: relu1_2, relu2_2, relu3_3)
- Computes L1 distance between feature maps
- Use as optional additional loss for generator

Create a `AdversarialLoss` class:
- Wrapper around BCEWithLogitsLoss
- Has methods for discriminator_loss(real_pred, fake_pred) and generator_loss(fake_pred)

---

## 7. Create `src/utils/metrics.py`

**Instructions for Copilot:**

Create functions:
- `compute_accuracy(predictions, labels)`: binary accuracy
- `compute_precision_recall_f1(predictions, labels)`: returns dict with precision, recall, f1
- `compute_confusion_matrix(predictions, labels)`: returns TP, TN, FP, FN
- `compute_roc_auc(predictions, labels)`: ROC-AUC score

Create a `MetricsTracker` class:
- Stores metrics across batches
- Has methods: update(metrics_dict), get_average(), reset()
- Pretty print method for logging

---

## 8. Create `src/utils/visualization.py`

**Instructions for Copilot:**

Create functions:
- `visualize_dct_features(images, save_path)`: 
  - Shows original image and its DCT representation side-by-side
  - Use matplotlib to plot
  
- `visualize_adversarial_examples(original, adversarial, perturbation, save_path)`:
  - Create 3-column grid: original, perturbation (amplified), adversarial
  - Save to file

- `plot_training_curves(metrics_history, save_path)`:
  - Plot discriminator loss, generator loss, accuracies over epochs
  - Save figure

- `visualize_feature_maps(model, image, save_path)`:
  - Extract and visualize intermediate feature maps from discriminator
  - Use hooks to capture activations

---

## 9. Create `config.py`

**Instructions for Copilot:**

Create a configuration class or dictionary with:
- Dataset settings: dataset_name, cache_dir, num_workers
- Model settings: pretrained, epsilon (perturbation strength)
- Training settings: batch_size, epochs, d_lr, g_lr
- Paths: checkpoint_dir, log_dir, output_dir
- Device: 'cuda' if available
- Mixed precision: use_amp (boolean)
- Logging: wandb_project, log_interval, save_interval

Use dataclasses or a simple dict structure.

---

## 10. Create `train.py`

**Instructions for Copilot:**

Create main training script:
- Import all necessary modules
- Load config
- Set random seeds for reproducibility (torch, numpy, random)
- Enable cudnn benchmark
- Load dataset using load_dataset function
- Apply preprocessing with transforms
- Create DataLoaders with collate_fn, num_workers, pin_memory=True, persistent_workers=True
- Instantiate discriminator and generator
- Create gpu_transform and compile it
- Instantiate DeepfakeGANTrainer
- Initialize wandb logging (optional)
- Create MetricsTracker
- Main training loop:
  - For each epoch:
    - Train loop: iterate batches, apply gpu_transform, call train_step
    - Update metrics tracker
    - Log metrics every log_interval batches
    - Validation loop every val_interval epochs
    - Save checkpoint every save_interval epochs
    - Update learning rate schedulers
    - Save best model based on validation accuracy
- At the end, save final checkpoint and plot training curves

---

## 11. Create `evaluate.py`

**Instructions for Copilot:**

Create evaluation script:
- Load trained discriminator from checkpoint
- Load test/validation dataset
- Compute metrics: accuracy, precision, recall, F1, ROC-AUC
- Generate classification report
- Visualize:
  - Confusion matrix
  - ROC curve
  - Sample predictions (correct and incorrect)
  - DCT features of misclassified examples
- Save results to JSON file
- Print summary statistics

---
## Additional Instructions

### Code Style Guidelines for Copilot:
- Use type hints for all function parameters and returns
- Add docstrings to all classes and functions (Google style)
- Use meaningful variable names
- Add comments for complex operations (especially DCT math)
- Handle edge cases (empty batches, mismatched sizes)
- Add input validation where appropriate
- Use torch.amp for mixed precision if use_amp is enabled
- Add progress bars with tqdm for long operations
- Log important information (model architecture, training progress)

### Error Handling:
- Wrap file I/O in try-except blocks
- Validate tensor shapes at critical points
- Add assertions for expected dimensions
- Gracefully handle CUDA out-of-memory errors

### Optimization:
- Use non_blocking=True for GPU transfers
- Enable torch.backends.cudnn.benchmark
- Use gradient accumulation if batch size is too large
- Implement gradient clipping (max_norm=1.0)

### Testing:
- Add a simple test to verify DCT implementation correctness
- Test that discriminator accepts [B,3,224,224] input
- Test that generator outputs correct shapes
- Verify training step doesn't crash with small batch

---

## Example Usage After Implementation

```python
# Train the model
python train.py --config config.py --dataset_name "your_dataset_name" --epochs 50

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth --split val
```

---

## Debugging Checklist for Copilot:

When implementing, ensure:
1. ✅ DCT output is [B, 1, 224, 224] for grayscale
2. ✅ Discriminator accepts RGB [B, 3, 224, 224] and outputs [B, 1]
3. ✅ Generator outputs (adversarial_images, perturbation)
4. ✅ All models move to correct device
5. ✅ Gradients flow correctly (no detach in wrong places)
6. ✅ Loss values are reasonable (not NaN or infinity)
7. ✅ Batch splitting handles cases where all samples are real or all fake
8. ✅ Skip connections in U-Net have matching dimensions
9. ✅ torch.compile() is called after model initialization
10. ✅ Data augmentation (if any) preserves label correctness

---

This specification should allow GitHub Copilot to generate a complete, working implementation. Each section is detailed enough that Copilot can generate correct code while maintaining consistency across the project.