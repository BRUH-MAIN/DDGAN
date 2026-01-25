# Deepfake Detection GAN (DDGAN)

A GAN-based deepfake detector that leverages DCT (Discrete Cosine Transform) frequency analysis and adversarial training for robust deepfake detection.

## Features

- **DCT-based Feature Extraction**: Analyzes frequency domain artifacts left by deepfake generation
- **Adversarial Training**: Generator creates perturbations to make the discriminator more robust
- **ConvNeXt Backbone**: Uses pretrained ConvNeXt-Tiny for powerful feature extraction
- **PyTorch Lightning**: Efficient, scalable training with automatic mixed precision
- **HuggingFace Integration**: Easy dataset loading from the HuggingFace Hub
- **FaceForensics++ Support**: Native support for FF++ image dataset
- **Data Imbalance Handling**: Multiple strategies including Focal Loss, AAML, and weighted sampling

## Architecture

### Discriminator
- DCT Feature Extractor: Extracts frequency-domain features
- ConvNeXt-Tiny backbone (pretrained on ImageNet)
- Classification head with dropout regularization

### Generator (U-Net)
- Encoder-decoder architecture with skip connections
- Frequency-aware bottleneck for DCT-domain processing
- Outputs bounded perturbations (controlled by epsilon)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DDGAN.git
cd DDGAN

# Install dependencies
pip install -e .
```

## Quick Start

### Training with HuggingFace Dataset

```bash
# Basic training
python train.py --dataset_name "your-dataset/name" --epochs 50

# Training with custom parameters
python train.py \
    --dataset_name "your-dataset/name" \
    --batch_size 32 \
    --epochs 100 \
    --precision "16-mixed"
```

### Training with FaceForensics++ Dataset

```bash
# Basic training with focal loss (recommended for imbalanced data)
python train_ff.py --data_dir ./images_dataset --epochs 50 --loss_type focal

# Training with AAML (Additive Angular Margin Loss)
python train_ff.py --data_dir ./images_dataset --loss_type aaml

# Combined approach for best performance
python train_ff.py --data_dir ./images_dataset --loss_type combined

# Full configuration example
python train_ff.py \
    --data_dir ./images_dataset \
    --batch_size 32 \
    --epochs 100 \
    --loss_type focal \
    --focal_gamma 2.0 \
    --focal_alpha 0.25 \
    --precision "16-mixed"
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py \
    --checkpoint checkpoints/best-epoch=XX-val_f1=0.XXXX.ckpt \
    --dataset_name "your-dataset/name"
```

## Data Imbalance Handling

The FaceForensics++ dataset has severe class imbalance (6:1 fake:real ratio). We implement multiple strategies:

### Available Loss Functions

| Loss Type | Description | Use Case |
|-----------|-------------|----------|
| `bce` | Standard BCE | Baseline |
| `focal` | Focal Loss | General imbalance |
| `weighted_bce` | Class-weighted BCE | Simple weighting |
| `aaml` | Angular Margin Loss | Better feature discrimination |
| `combined` | Focal + AAML | Best performance |

### Strategies Implemented

1. **Weighted Random Sampling**: Ensures balanced batches without losing data
2. **Focal Loss**: Down-weights easy examples, focuses on hard ones
3. **AAML (Additive Angular Margin Loss)**: Improves feature discrimination
4. **Class Weighting**: Compensates for imbalanced class distribution

See [docs/DATA_IMBALANCE.md](docs/DATA_IMBALANCE.md) for detailed documentation.

### Command Line Arguments

#### train_ff.py (FaceForensics++)
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | ./images_dataset | Path to FF++ images |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 50 | Training epochs |
| `--loss_type` | focal | Loss function type |
| `--focal_gamma` | 2.0 | Focal loss gamma |
| `--focal_alpha` | 0.25 | Focal loss alpha |
| `--aaml_margin` | 0.5 | AAML margin |
| `--no_weighted_sampling` | False | Disable weighted sampling |

## Project Structure

```
DDGAN/
├── main.py              # Entry point
├── train.py             # Training script (HuggingFace datasets)
├── train_ff.py          # Training script (FaceForensics++)
├── evaluate.py          # Evaluation script
├── config.py            # Configuration classes
├── pyproject.toml       # Project dependencies
├── docs/
│   └── DATA_IMBALANCE.md # Data imbalance documentation
├── images_dataset/      # FaceForensics++ images
│   ├── image_dataset_metadata.csv
│   ├── original/        # Real images
│   ├── Deepfakes/       # Fake images
│   ├── Face2Face/       # Fake images
│   └── ...
├── src/
│   ├── data/
│   │   ├── dataset.py   # HuggingFace dataset loading
│   │   ├── ff_dataset.py # FaceForensics++ dataset
│   │   └── transforms.py # Image transforms
│   ├── models/
│   │   ├── dct_extractor.py  # DCT feature extraction
│   │   ├── discriminator.py  # DCT Discriminator
│   │   └── generator.py      # U-Net Generator
│   ├── training/
│   │   ├── trainer.py   # Lightning training module
│   │   ├── ff_trainer.py # FF++ trainer with imbalance handling
│   │   ├── losses.py    # Base loss functions
│   │   └── imbalance_losses.py # Focal, AAML, weighted losses
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       └── visualization.py # Plotting utilities
├── checkpoints/         # Saved models
└── logs/               # Training logs
```

## Training Workflow

1. **Data Loading**: Images loaded from HuggingFace Hub with on-the-fly transforms
2. **Discriminator Training**:
   - Real images: labeled as real (1)
   - Fake images: labeled as fake (0)
   - Perturbed images: treated as fake to improve robustness
3. **Generator Training**:
   - Creates perturbations to fool the discriminator
   - Adversarial + perceptual loss
4. **Validation**: Computes accuracy, F1, ROC-AUC on held-out data

## Checkpoints

Lightning automatically saves:
- `best-epoch=XX-val_accuracy=0.XXXX.ckpt`: Best model by validation accuracy
- `checkpoint-epoch=XX.ckpt`: Periodic checkpoints (every 5 epochs)
- `last.ckpt`: Latest model state

## Logging

Training logs are saved as CSV files in the `logs/` directory:
- `logs/deepfake_gan/version_X/metrics.csv`: All training metrics
- `logs/deepfake_gan/version_X/hparams.yaml`: Hyperparameters

## Dataset Format

The dataset should be in HuggingFace format with:
- `image`: PIL Image or path to image
- `label`: 0 for fake, 1 for real

Example datasets:
- Custom dataset uploaded to HuggingFace Hub
- Local datasets loaded via `datasets` library

## Requirements

- Python 3.9+
- PyTorch 2.1+
- PyTorch Lightning 2.5+
- CUDA-capable GPU (recommended)

See [pyproject.toml](pyproject.toml) for full dependency list.

## License

MIT License
