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
echo "[1/7] Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. Training will be slow!"
fi
echo ""

# Check Python version
echo "[2/7] Checking Python version..."
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
echo "[3/7] Installing system dependencies..."
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

# Check if uv is installed
echo "[4/7] Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "⚠ uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv detected"
fi
echo ""

# Install dependencies with uv
echo "[5/7] Installing dependencies with uv..."
echo "This may take several minutes..."

# CRITICAL: Install PyTensor from parent directory in EDITABLE mode FIRST
# This ensures the local development version is used, not PyPI
echo "Installing PyTensor from source (editable)..."
cd ../../../
uv pip install -e ".[development,onnx]"

# Verify PyTensor is installed from source
echo "Verifying PyTensor installation..."
PYTENSOR_LOCATION=$(uv pip show pytensor | grep "Location:" | awk '{print $2}')
if [[ "$PYTENSOR_LOCATION" == *"site-packages"* ]]; then
    echo "❌ ERROR: PyTensor was installed from PyPI, not from source!"
    echo "   Location: $PYTENSOR_LOCATION"
    echo "   Expected: /path/to/pytensor (source directory)"
    exit 1
fi
echo "✓ PyTensor installed from source: $PYTENSOR_LOCATION"

# Return to demo directory
cd examples/onnx/onnx-yolo-demo/

# Install demo dependencies WITHOUT uv sync (which would override editable PyTensor)
# Using uv pip install respects already-installed editable packages
echo "Installing YOLO demo dependencies..."
uv pip install -e .

# Note: We do NOT use "uv sync" here because:
# - uv sync reads/creates uv.lock with pinned versions from PyPI
# - This would reinstall pytensor from PyPI, overriding our editable install
# - uv pip install respects already-installed packages (including editable ones)

echo "✓ All Python packages installed"
echo ""

# Create data directory
echo "[6/7] Setting up directories and configuration..."
mkdir -p data/coco
mkdir -p checkpoints
mkdir -p logs
echo "✓ Directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cat > .env << 'ENVEOF'
# PyTensor Configuration
PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"

# JAX Platform Configuration - FORCE GPU USAGE
JAX_PLATFORMS="cuda"
JAX_ENABLE_X64=False

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
echo "[7/7] Cleaning Python cache..."
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
echo "1. Login to Weights & Biases (for training visualization):"
echo "   wandb login"
echo "   (You'll need your API key from https://wandb.ai/authorize)"
echo ""
echo "2. Run the training script:"
echo "   bash scripts/train.sh"
echo ""
echo "3. To run training in background and logout:"
echo "   nohup bash scripts/train.sh > training.log 2>&1 &"
echo "   # Monitor with: tail -f training.log"
echo ""
echo "4. After training completes, download the ONNX model:"
echo "   scp user@server:$(pwd)/checkpoints/yolo11n_best.onnx ."
echo ""
echo "5. To run tests:"
echo "   uv run pytest tests/ -v"
echo "   # Or with specific profile: HYPOTHESIS_PROFILE=dev uv run pytest tests/ -v"
echo ""
echo "=========================================="
