#!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name=multilingual-transformer-1
#SBATCH --nodelist=asaicomputenode03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/dist_home/nooglers/nooglers/Roshan/NLP-MultiHop-Q-A/logs/%x_%j.out
#SBATCH --error=/dist_home/nooglers/nooglers/Roshan/NLP-MultiHop-Q-A/logs/%x_%j.err

set -e

# =====================================================
# CONFIGURATION - Modify these paths for your setup
# =====================================================
BASE_DIR="/dist_home/nooglers/nooglers/Roshan/NLP-MultiHop-Q-A"
LOG_DIR="${BASE_DIR}/logs"
CONDA_ENV="new"  # Your conda environment name

# Training parameters
BATCH_SIZE=64
GRAD_ACCUM=4
EPOCHS=10
GPU_ID="0,1"  # Use GPU 1 (set to 0 for GPU 0, or "0,1" for both)

# =====================================================
# SETUP
# =====================================================

# Create log directory
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "MULTILINGUAL TRANSFORMER TRAINING"
echo "=============================================="
echo "Start time: $(date)"
echo "Base dir: ${BASE_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accumulation: ${GRAD_ACCUM}"
echo "=============================================="

# Load CUDA module (safe - won't exit if missing)
module purge || true
module load cuda || echo "CUDA module not found, relying on system CUDA"

# Set environment variables

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Change to project directory
cd "${BASE_DIR}"

# Activate virtual environment
VENV_PATH="/dist_home/nooglers/nooglers/Roshan/new/bin/activate"
if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
    echo "Activated virtual environment: ${VENV_PATH}"
else
    echo "ERROR: Virtual environment not found at ${VENV_PATH}" >&2
    exit 1
fi

# Print Python and GPU info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# =====================================================
# TRAINING PIPELINE
# =====================================================

# Step 3: Train Model
echo ""
echo "=============================================="
echo "[3/3] TRAINING MODEL"
echo "=============================================="
python 3_train_model.py \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --epochs ${EPOCHS} \
    2>&1 | tee "${LOG_DIR}/3_train_model_${SLURM_JOB_ID}.log"

# =====================================================
# COMPLETION
# =====================================================

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "End time: $(date)"
echo "=============================================="
echo "Model saved to: ${BASE_DIR}/transformer_model_output/"
echo "Logs saved to: ${LOG_DIR}/"
echo "=============================================="