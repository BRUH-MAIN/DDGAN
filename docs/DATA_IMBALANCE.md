# Data Imbalance Handling for FaceForensics++ Dataset

## Overview

This document details the implementation of data imbalance handling strategies for the FaceForensics++ deepfake detection dataset. The dataset contains significantly more fake samples than real samples, which can lead to biased models that struggle to correctly identify real images.

## Dataset Analysis

### FaceForensics++ Structure

The dataset contains images extracted from videos in the following categories:

| Category | Type | Description |
|----------|------|-------------|
| original | REAL | Original unmanipulated videos |
| Deepfakes | FAKE | DeepFake manipulation |
| Face2Face | FAKE | Face reenactment |
| FaceSwap | FAKE | Face swapping |
| FaceShifter | FAKE | Advanced face swapping |
| NeuralTextures | FAKE | Neural texture manipulation |
| DeepFakeDetection | FAKE | Additional deepfakes |

### Imbalance Characteristics

With 6 fake categories and only 1 real category, the dataset exhibits a severe class imbalance (approximately 6:1 fake to real ratio). This imbalance can cause:

1. **Biased predictions**: Model tends to predict "fake" more often
2. **Poor recall for real images**: Minority class gets ignored
3. **Misleading accuracy**: High accuracy can mask poor minority class performance
4. **Gradient domination**: Majority class dominates training gradients

## Implemented Solutions

We implement a **hybrid approach** combining multiple strategies, as research shows this yields the best results for imbalanced classification.

### 1. Data-Level Approaches

#### Weighted Random Sampling

Instead of undersampling (losing data) or SMOTE (synthetic data), we use weighted random sampling to ensure balanced batch composition during training.

```python
# In FaceForensicsDataset
def get_weighted_sampler(self) -> WeightedRandomSampler:
    """Oversample minority class (real) during training."""
    # Inverse frequency weighting
    class_weights = total / (num_classes * class_counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples, replacement=True)
```

**Advantages:**
- No data loss (unlike undersampling)
- No synthetic artifacts (unlike SMOTE)
- Ensures balanced batches during training
- Real data is reused, not artificially generated

### 2. Algorithm-Level Approaches

#### A. Focal Loss

Focal Loss addresses class imbalance by down-weighting well-classified examples and focusing on hard, misclassified ones.

**Mathematical Formulation:**
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ = probability of correct class
- $\alpha_t$ = class balancing weight
- $\gamma$ = focusing parameter (typically 2.0)

**Key Properties:**
- When $\gamma = 0$, equivalent to cross-entropy
- Higher $\gamma$ = more focus on hard examples
- $\alpha$ provides class balancing

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (alpha_t * focal_weight * bce).mean()
```

#### B. Weighted Binary Cross-Entropy

Simple but effective: apply higher loss weight to minority class errors.

**Mathematical Formulation:**
$$L_{WBCE} = -w_{pos} \cdot y \cdot \log(\hat{y}) - w_{neg} \cdot (1-y) \cdot \log(1-\hat{y})$$

**Implementation:**
```python
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        self.pos_weight = pos_weight  # e.g., 5.0 if fake:real = 5:1
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
```

#### C. AAML (Additive Angular Margin Loss)

Adapted from ArcFace for deepfake detection. This loss embeds features on a hypersphere and adds angular margin to increase class separability.

**Mathematical Formulation:**
$$L_{AAML} = -\log\frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + e^{s \cdot \cos(\theta_j)}}$$

Where:
- $s$ = scaling factor (typically 30-64)
- $m$ = angular margin (typically 0.3-0.5)
- $\theta$ = angle between feature and class center

**Key Benefits:**
- Better feature discrimination
- Improved generalization to unseen deepfake methods
- More robust decision boundaries

**Implementation:**
```python
class AAMLoss(nn.Module):
    def __init__(self, in_features, scale=30.0, margin=0.5):
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(2, in_features))
    
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features, weight)
        
        # Apply angular margin to target class
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * cos_m - sine * sin_m
        
        # Scale and compute cross-entropy
        output = one_hot * phi + (1 - one_hot) * cosine
        output = output * self.scale
        return F.cross_entropy(output, labels)
```

#### D. Asymmetric Focal Loss

For extremely imbalanced cases, use different focusing parameters for each class.

```python
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0):
        self.gamma_pos = gamma_pos  # Less focus on easy positives
        self.gamma_neg = gamma_neg  # More focus on hard negatives
```

### 3. Combined Approach

Our recommended approach combines multiple strategies:

```python
class CombinedImbalanceLoss(nn.Module):
    def __init__(self, focal_alpha, focal_gamma, aaml_margin, aaml_weight=0.3):
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.aaml_loss = AAMLoss(margin=aaml_margin)
        self.aaml_weight = aaml_weight
    
    def forward(self, logits, targets, features):
        focal = self.focal_loss(logits, targets)
        aaml, _ = self.aaml_loss(features, targets)
        return (1 - self.aaml_weight) * focal + self.aaml_weight * aaml
```

## Class Weight Computation

We implement multiple methods for computing class weights:

### 1. Inverse Frequency (Standard)
$$w_c = \frac{N}{C \cdot n_c}$$

### 2. Square Root Inverse (Smoothed)
$$w_c = \sqrt{\frac{N}{C \cdot n_c}}$$

### 3. Effective Number (Best for Long-Tail)
$$w_c = \frac{1}{E_c} \text{ where } E_c = \frac{1 - \beta^{n_c}}{1 - \beta}$$

```python
def compute_class_weights(n_fake, n_real, method='effective'):
    if method == 'effective':
        beta = 0.9999
        eff_fake = (1 - beta ** n_fake) / (1 - beta)
        eff_real = (1 - beta ** n_real) / (1 - beta)
        w_fake = 1 / eff_fake
        w_real = 1 / eff_real
    return normalize([w_fake, w_real])
```

## Evaluation Metrics

For imbalanced data, accuracy alone is misleading. We use:

| Metric | Formula | Why Important |
|--------|---------|---------------|
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Balances precision and recall |
| **Balanced Accuracy** | $\frac{TPR + TNR}{2}$ | Equal weight to both classes |
| **MCC** | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Robust for imbalance |
| **Specificity** | $\frac{TN}{TN + FP}$ | Ability to detect fakes |
| **Recall** | $\frac{TP}{TP + FN}$ | Ability to detect real |

## Usage

### Training with FaceForensics++

```bash
# Basic training with focal loss and weighted sampling
python train_ff.py --data_dir ./images_dataset --loss_type focal

# Training with AAML loss
python train_ff.py --data_dir ./images_dataset --loss_type aaml --aaml_margin 0.5

# Combined approach (recommended)
python train_ff.py --data_dir ./images_dataset --loss_type combined

# Disable weighted sampling (not recommended)
python train_ff.py --data_dir ./images_dataset --no_weighted_sampling
```

### Available Loss Types

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `bce` | Standard BCE | Baseline |
| `focal` | Focal Loss | General imbalance |
| `weighted_bce` | Weighted BCE | Simple weighting |
| `aaml` | Angular Margin Loss | Better discrimination |
| `combined` | Focal + AAML | Best performance |

## Experimental Comparison

Based on research literature, expected performance comparison:

| Method | Accuracy | F1 Score | MCC |
|--------|----------|----------|-----|
| BCE (baseline) | ~85% | ~0.60 | ~0.45 |
| Weighted BCE | ~87% | ~0.70 | ~0.55 |
| Focal Loss | ~88% | ~0.75 | ~0.60 |
| AAML | ~89% | ~0.78 | ~0.65 |
| Combined + Sampling | ~91% | ~0.82 | ~0.70 |

*Note: Actual results depend on dataset split and hyperparameters.*

## References

1. Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
2. Deng, J., et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR.
3. Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR.
4. Rossler, A., et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV.

## File Structure

```
src/
├── data/
│   ├── ff_dataset.py      # FaceForensics++ dataset loader
│   └── transforms.py      # Image transforms
└── training/
    ├── ff_trainer.py      # Lightning module with imbalance handling
    └── imbalance_losses.py # Loss functions (Focal, AAML, etc.)

train_ff.py                # Training script for FF++ dataset
```
