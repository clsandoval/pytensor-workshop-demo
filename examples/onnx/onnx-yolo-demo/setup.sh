#!/bin/bash
# YOLO11n Training Environment Setup Script
# Run this once after cloning the repo on your GPU server
#
# Usage: bash setup.sh

set -e  # Exit on any error

# CRITICAL: Set PyTensor flags BEFORE any Python/PyTensor imports
export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run"

echo "=========================================="
echo "YOLO11n Training Environment Setup"
echo "=========================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux. You may need to modify it for other systems."
fi

# Check for GPU
echo "[1/8] Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. Training will be slow!"
fi
echo ""

# Check Python version
echo "[2/8] Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    python_version=$(python3.11 --version 2>&1 | awk '{print $2}')
    echo "✓ Using Python 3.11: $python_version"
else
    PYTHON_CMD=python3
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python version: $python_version"
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        echo "Error: Python 3.11+ required. Found: $python_version"
        echo "Install with: sudo apt install -y python3.11 python3.11-venv python3.11-dev"
        exit 1
    fi
fi
echo ""

# Install system dependencies
echo "[3/8] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    echo "Updating package list..."
    sudo apt-get update -qq
    echo "Installing build essentials and CUDA support..."
    if command -v python3.11 &> /dev/null; then
        sudo apt-get install -y build-essential python3.11-dev python3.11-venv git wget curl
    else
        sudo apt-get install -y build-essential python3-dev git wget curl
    fi
else
    echo "⚠ apt-get not found. Skipping system dependencies."
    echo "Please ensure you have: build-essential, python3-dev, git, wget, curl"
fi
echo ""

# Create virtual environment
echo "[4/8] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[5/8] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[6/8] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ pip upgraded"
echo ""

# Install PyTensor with JAX backend
echo "[7/8] Installing PyTensor and dependencies..."
echo "This may take several minutes..."

# Install from the parent pytensor directory
cd ../../../
pip install -e . --quiet

# Install JAX with CUDA support
# Note: Adjust cuda version if needed (cuda12 shown here)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet

# Install training dependencies
pip install --quiet \
    numpy \
    scipy \
    pillow \
    wandb \
    pycocotools \
    tqdm \
    pyyaml \
    requests

cd examples/onnx/onnx-yolo-demo/

echo "✓ All Python packages installed"
echo ""

# Create data directory
echo "[8/8] Setting up directories and configuration..."
mkdir -p data/coco
mkdir -p checkpoints
mkdir -p logs
echo "✓ Directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cat > .env << 'ENVEOF'
# PyTensor Configuration (JAX backend will auto-detect GPU)
PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run"

# JAX GPU Memory Configuration
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# WandB Configuration
WANDB_PROJECT=yolo11n-pytensor

# System Configuration
PYTHONUNBUFFERED=1
ENVEOF
    echo "✓ .env file created with defaults"
    echo "  Edit .env to customize settings (see .env.example for all options)"
else
    echo "✓ .env file already exists"
fi
echo ""

# Clean Python cache to ensure float32 config takes effect
echo ""
echo "Cleaning Python cache..."
find ../../../ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ../../../ -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Cache cleaned"

# Login to WandB (optional, can be done later)
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Login to Weights & Biases (for training visualization):"
echo "   wandb login"
echo "   (You'll need your API key from https://wandb.ai/authorize)"
echo ""
echo "3. Run the training script:"
echo "   bash train.sh"
echo ""
echo "4. To run training in background and logout:"
echo "   nohup bash train.sh > training.log 2>&1 &"
echo "   # Monitor with: tail -f training.log"
echo ""
echo "5. After training completes, download the ONNX model:"
echo "   scp user@server:$(pwd)/checkpoints/yolo11n_best.onnx ."
echo ""
echo "=========================================="
