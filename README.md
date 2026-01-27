# Deepfake Detection GAN

A robust deepfake detector using GAN-based adversarial training with DCT features and ConvNeXt backbone.

## 🎯 Project Overview

This project implements a novel approach to deepfake detection by combining:
- **DCT (Discrete Cosine Transform)** features for frequency-domain analysis
- **ConvNeXt** backbone with pretrained ImageNet weights
- **U-Net Generator** for adversarial perturbations
- **Adversarial robustness training** for better generalization

## 🏗️ Architecture

### Discriminator
- **Input**: RGB images (224×224)
- **Feature Extraction**: DCT transformation to grayscale frequency features
- **Backbone**: ConvNeXt-Tiny (27.8M parameters, pretrained on ImageNet)
- **Output**: Binary classification (real/fake)

### Generator
- **Architecture**: U-Net with Frequency-Aware Bottleneck
- **Purpose**: Generate adversarial perturbations to make discriminator more robust
- **Output**: Perturbed images with bounded epsilon (default: 0.03)

## 📊 Dataset

**CelebDF-v2** preprocessed into image format:
```
celebdfv2_images/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

See [preprocess_celebdfv2.ipynb](preprocess_celebdfv2.ipynb) for preprocessing details.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd DDGAN

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Dataset (if needed)

```bash
# Run preprocessing notebook
jupyter notebook preprocess_celebdfv2.ipynb
```

### 3. Train Model

```bash
# Basic training (with default settings)
python train.py

# Custom training
python train.py \
    --data-dir celebdfv2_images \
    --batch-size 32 \
    --epochs 50 \
    --lr 2e-4 \
    --devices 2
```

### 4. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs
```

## 📁 Project Structure

```
DDGAN/
├── config.py                   # Configuration management
├── train.py                    # Main training script
├── lightning_module.py         # PyTorch Lightning module
├── requirements.txt            # Python dependencies
├── preprocess_celebdfv2.ipynb  # Dataset preprocessing
├── models/
│   ├── __init__.py
│   ├── dct_extractor.py       # DCT feature extractor
│   ├── discriminator.py       # Discriminator model
│   └── generator.py           # Generator model
├── data/
│   ├── __init__.py
│   └── datamodule.py          # PyTorch Lightning DataModule
├── checkpoints/               # Model checkpoints (created during training)
└── logs/                      # TensorBoard logs (created during training)
```

## 🔧 Configuration

Edit [config.py](config.py) to customize:
- Data parameters (batch size, augmentations)
- Model architecture (backbone, channels)
- Training hyperparameters (learning rate, epochs)
- Hardware settings (GPUs, precision)

## 📈 Training Details

### Adversarial Robustness Training

Unlike traditional GAN training, this approach:
1. **Discriminator**: Learns to classify real/fake while being robust to perturbations
2. **Generator**: Creates adversarial perturbations to test discriminator
3. **Goal**: Discriminator that generalizes well to unseen deepfakes

### Loss Functions

**Discriminator Loss**:
```
L_D = L_real + L_fake + 0.5 × L_adv
```

**Generator Loss**:
```
L_G = L_adv + 0.1 × L_perturb
```

### Optimization

- **Optimizer**: AdamW (lr=2e-4, betas=(0.5, 0.999))
- **Scheduler**: CosineAnnealingLR
- **Precision**: Mixed (16-bit)
- **Gradient Clipping**: max_norm=1.0

## 📊 Expected Results

**Target Performance**:
- Accuracy: > 80%
- Precision: > 75%
- Recall: > 75%
- F1 Score: > 75%
- ROC-AUC: > 85%

**Training Time**:
- ~10-15 min/epoch on 2× T4 GPUs
- Total: ~8-12 hours for 50 epochs

## 🧪 Testing Components

Test individual components:

```bash
# Test DCT extractor
python -m models.dct_extractor

# Test Discriminator
python -m models.discriminator

# Test Generator
python -m models.generator

# Test DataModule
python -m data.datamodule

# Test Lightning module
python lightning_module.py
```

## 📝 Key Features

1. **DCT Features**: Frequency-domain analysis captures deepfake artifacts
2. **Transfer Learning**: Pretrained ConvNeXt weights accelerate training
3. **Adversarial Training**: Generator improves discriminator robustness
4. **Mixed Precision**: 2-3× speedup with AMP
5. **PyTorch Lightning**: Professional training pipeline with DDP support

## 🔬 Research Background

This implementation is based on insights from:
- Frequency-domain deepfake detection research
- Adversarial robustness training techniques
- Modern CNN architectures (ConvNeXt)
- U-Net for image-to-image translation

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Add your contact information here]

---

**Note**: This project is for research and educational purposes. Use responsibly and ethically.
