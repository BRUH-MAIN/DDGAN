# Deepfake Detection GAN (DDGAN)

A GAN-based deepfake detector that leverages DCT (Discrete Cosine Transform) frequency analysis and adversarial training for robust deepfake detection.

## Features

- **DCT-based Feature Extraction**: Analyzes frequency domain artifacts left by deepfake generation
- **Adversarial Training**: Generator creates perturbations to make the discriminator more robust
- **ConvNeXt Backbone**: Uses pretrained ConvNeXt-Tiny for powerful feature extraction
- **PyTorch Lightning**: Efficient, scalable training with automatic mixed precision
- **HuggingFace Integration**: Easy dataset loading from the HuggingFace Hub

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

### Training

```bash
# Basic training
python train.py --dataset_name "your-dataset/name" --epochs 50

# Training with custom parameters
python train.py \
    --dataset_name "your-dataset/name" \
    --batch_size 32 \
    --epochs 100 \
    --d_lr 2e-4 \
    --g_lr 2e-4 \
    --precision "16-mixed" \
    --checkpoint_dir "checkpoints" \
    --log_dir "logs"

# Fast development run
python train.py --dataset_name "your-dataset/name" --fast_dev_run
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py \
    --checkpoint checkpoints/best-epoch=XX-val_accuracy=0.XXXX.ckpt \
    --dataset_name "your-dataset/name" \
    --split val \
    --output_dir evaluation_results
```

### Command Line Arguments

#### train.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | None | HuggingFace dataset name |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 50 | Number of training epochs |
| `--d_lr` | 2e-4 | Discriminator learning rate |
| `--g_lr` | 2e-4 | Generator learning rate |
| `--precision` | "16-mixed" | Training precision (32, 16-mixed, bf16-mixed) |
| `--epsilon` | 0.03 | Perturbation strength |
| `--checkpoint_dir` | "checkpoints" | Directory for checkpoints |
| `--log_dir` | "logs" | Directory for logs |
| `--resume` | None | Path to checkpoint to resume from |
| `--fast_dev_run` | False | Run a fast development test |

#### evaluate.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--dataset_name` | None | HuggingFace dataset name |
| `--split` | "val" | Dataset split (train/val/test) |
| `--batch_size` | 32 | Evaluation batch size |
| `--output_dir` | "evaluation_results" | Output directory |
| `--no_viz` | False | Skip visualization generation |

## Project Structure

```
DDGAN/
├── main.py              # Entry point
├── train.py             # Training script (Lightning)
├── evaluate.py          # Evaluation script
├── config.py            # Configuration classes
├── pyproject.toml       # Project dependencies
├── src/
│   ├── data/
│   │   ├── dataset.py   # Dataset loading
│   │   └── transforms.py # Image transforms
│   ├── models/
│   │   ├── dct_extractor.py  # DCT feature extraction
│   │   ├── discriminator.py  # DCT Discriminator
│   │   └── generator.py      # U-Net Generator
│   ├── training/
│   │   ├── trainer.py   # Lightning training module
│   │   └── losses.py    # Loss functions
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
