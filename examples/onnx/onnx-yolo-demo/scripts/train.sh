#!/bin/bash
# YOLO11n Training Script
# This script runs the training with sensible defaults
#
# Usage: bash train.sh

set -e  # Exit on error

echo "=========================================="
echo "YOLO11n Training"
echo "=========================================="
echo ""

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    echo "✓ Environment variables loaded"
else
    echo "⚠ No .env file found. Using system defaults."
    echo "  Run setup.sh to create default .env file."
fi
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found. Run setup.sh first!"
    exit 1
fi

# Check WandB login status (non-interactive)
echo ""
echo "Checking WandB login status..."
if wandb status 2>&1 | grep -q "Logged in"; then
    echo "✓ WandB is configured"
    NO_WANDB=""
else
    echo "⚠ WandB not logged in. Running without WandB logging."
    echo "  To enable WandB next time, run: wandb login"
    NO_WANDB="--no-wandb"
fi

# Check for GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    GPU_AVAILABLE=1
else
    echo "⚠ No GPU detected. Training will be very slow!"
    GPU_AVAILABLE=0
fi

# Training configuration
EPOCHS=100
BATCH_SIZE=8
LR=0.01
IMAGE_SIZE=320
NUM_CLASSES=2

# Adjust batch size based on GPU memory
if [ $GPU_AVAILABLE -eq 1 ]; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ $GPU_MEM -lt 8000 ]; then
        BATCH_SIZE=4
        echo "⚠ GPU has less than 8GB memory. Reducing batch size to 4"
    elif [ $GPU_MEM -gt 16000 ]; then
        BATCH_SIZE=16
        echo "✓ GPU has sufficient memory. Increasing batch size to 16"
    fi
fi

echo ""
echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Number of classes: $NUM_CLASSES"
echo ""

# Create necessary directories
mkdir -p data/coco
mkdir -p checkpoints
mkdir -p logs

# Download COCO dataset if not present
echo ""
echo "Checking for COCO dataset..."
if [ ! -f "data/coco/annotations/instances_train2017.json" ]; then
    echo "COCO dataset not found. Downloading..."
    echo "⚠ This will download ~20GB of data. It may take 30-60 minutes."
    echo ""
    python dataset.py --data-dir ./data/coco --split train
    echo "✓ COCO dataset downloaded"
else
    echo "✓ COCO dataset found"
fi
echo ""

# Start training
echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

# CRITICAL: Set PyTensor flags BEFORE Python starts
# This must happen in the shell, not inside Python
export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run"
export JAX_PLATFORMS="cuda,cpu"
export JAX_ENABLE_X64=False
echo "PyTensor config: $PYTENSOR_FLAGS"
echo "JAX platforms: $JAX_PLATFORMS"
echo ""

python train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --image-size $IMAGE_SIZE \
    --num-classes $NUM_CLASSES \
    --checkpoint-dir ./checkpoints \
    --save-every 10 \
    --log-every 10 \
    --export-onnx \
    $NO_WANDB

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Checkpoints: ./checkpoints/"
echo "  Best model: ./checkpoints/best_model.npz"
echo "  ONNX model: ./checkpoints/yolo11n_best.onnx"
echo ""
echo "To download the ONNX model to your laptop:"
echo "  scp user@server:$(pwd)/checkpoints/yolo11n_best.onnx ."
echo ""
echo "=========================================="
