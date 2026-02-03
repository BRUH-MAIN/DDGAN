# Deepfake Robustness Training (DDGAN)

This project trains a deepfake detector for **adversarial robustness**, not classical GAN realism. The generator produces **bounded perturbations** and the discriminator remains a **strong classifier** that is encouraged to be consistent under those perturbations.

## Summary

- **Discriminator:** ConvNeXt backbone with a **dual-stream** RGB + DCT fusion (default).
- **Generator:** U-Net that outputs bounded perturbations (epsilon-bounded).
- **Objective:** Robust classification via **consistency loss** and a **margin-based generator loss** (no BCE on adversarial labels).

## Label Convention (Important)

This codebase normalizes labels to:

- **real = 1 (positive class)**
- **fake = 0 (negative class)**

This applies to both local folders and HuggingFace datasets and is used consistently across training and validation.

## Losses (Implemented)

Let $D(x)$ be discriminator logits and $G(x)$ be the perturbation, $x_{adv} = x + G(x)$.

**Discriminator loss:**

$$
L_D = L_{real} + L_{fake} + w_{cons}(t) \cdot \lambda_{cons} \cdot \mathbb{E}\left[(\sigma(D(x)) - \sigma(D(x_{adv})))^2\right]
$$

$w_{cons}(t)$ ramps from $0$ to $1$ over the first few epochs to avoid early collapse.

**Generator loss (margin + perturbation):**

$$
L_G = \mathbb{E}\left[\frac{\max(0, D(x_{adv}) - m)}{\overline{|D(x_{adv})|}+\epsilon}\right] + \lambda_{pert} \cdot \mathbb{E}[||\delta||_2^2]
$$

Where $m$ is the margin and $\delta = G(x)$ is epsilon-bounded.

## Metrics (Logged)

Validation reports **both** clean and adversarial AUC:

- `val/auc_clean`
- `val/auc_adv`
- `val/robustness_gap = auc_clean - auc_adv`

Checkpoint selection and early stopping monitor **adversarial AUC** by default.

## Architecture

### Discriminator

- Dual-stream ConvNeXt: RGB stream + DCT stream with learnable frequency filters.
- Legacy single-stream DCT-only discriminator is supported.

### Generator

- U-Net with a frequency-aware bottleneck (explicit DCT/IDCT).
- Outputs epsilon-bounded perturbations.

## Configuration

You can configure training in two ways:

1. **Edit defaults** in [config.py](config.py)
2. **Override via CLI** in [train.py](train.py)

Key settings include:

- Model type (`d_type`, `d_fusion_type`)
- Loss weights (`consistency_weight`, `margin`, `perturb_weight`)
- Optimization (`learning_rate`, `d_lr_mult`, `g_lr_mult`)
- Hardware (`devices`, `precision`, `strategy`)

## Dataset

Supported sources:

- **Local folders**: `celebdfv2_images/{train,test}/{real,fake}`
- **HuggingFace**: set `--hf-dataset` (overrides local path)

See [preprocess_celebdfv2.ipynb](preprocess_celebdfv2.ipynb) for preprocessing details.

## Training

```bash
python train.py
```

Common overrides:

```bash
python train.py \
    --hf-dataset RohanRamesh/celebdfv2_224 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --devices 2 \
    --precision 16-mixed
```

## Monitoring

```bash
tensorboard --logdir logs
```

## Project Structure

```
DDGAN/
├── config.py                   # Configuration management
├── train.py                    # Main training script (CLI overrides)
├── lightning_module.py         # PyTorch Lightning module
├── models/
│   ├── dct_extractor.py         # DCT feature extractor
│   ├── discriminator.py         # Discriminator model(s)
│   └── generator.py             # Generator model
├── data/
│   └── datamodule.py            # DataModule + label normalization
└── logs/                        # TensorBoard logs
```

## Notes

- This is **not** GAN-style realism training.
- The generator does **not** synthesize fake images.
- Robustness should be judged using **adversarial AUC** and **robustness gap**.
