# YOLO11n GPU Demo (Lambda 22.04 + CUDA)

GPU-optimized version of the YOLO11n object detection demo using PyTensor, JAX with CUDA, and ONNX export.

## Key Differences from CPU Version

- **JAX Backend**: Uses `jax[cuda]` instead of `jax[cpu]`
- **Target Platform**: Optimized for Lambda Stack 22.04 with NVIDIA GPUs
- **Training Speed**: 10-20x faster on modern GPUs (RTX 3090, A100, etc.)
- **Dependencies**: CUDA-enabled JAX installation

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions.

```bash
# 1. Clone and navigate
cd pytensor/examples/onnx/onnx-yolo-demo-gpu

# 2. Run setup
bash scripts/setup.sh

# 3. Start training
bash scripts/train.sh
```

## System Requirements

### Required
- Ubuntu 22.04 (Lambda Stack recommended)
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- CUDA 11.8+ or 12.x
- Python 3.11+
- 50GB+ free disk space (for COCO dataset)

### Recommended
- 16GB+ GPU memory for larger batch sizes
- 32GB+ system RAM
- NVMe SSD for faster data loading

## Installation

### 1. Verify CUDA Setup

```bash
# Check GPU
nvidia-smi

# Verify CUDA version
nvcc --version

# Should show CUDA 11.8 or 12.x
```

### 2. Run Setup Script

```bash
bash scripts/setup.sh
```

This will:
- Install system dependencies
- Install UV package manager
- Install PyTensor from source (editable mode)
- Install JAX with CUDA support
- Install all demo dependencies
- Create directories and configuration files

### 3. Verify JAX CUDA Installation

```bash
uv run python -c "import jax; print(jax.devices())"
# Should output: [CudaDevice(id=0)]
```

## Training

### Quick Training

```bash
bash scripts/train.sh
```

### Custom Training

```bash
uv run python train.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.01 \
    --image-size 320 \
    --num-classes 2 \
    --checkpoint-dir ./checkpoints \
    --export-onnx
```

### Training Configuration

Edit `scripts/train.sh` to adjust:
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size (default: 8, increase for larger GPUs)
- `LR`: Learning rate (default: 0.01)
- `IMAGE_SIZE`: Input image size (default: 320)

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run GPU-specific tests
uv run pytest tests/ -v -m gpu

# Test JAX CUDA backend
uv run pytest tests/test_jax_backend.py -v

# Test model on GPU
uv run pytest tests/test_model_integration.py -v
```

## Performance

### Expected Training Times (100 epochs)

| GPU            | VRAM  | Time      | Batch Size |
|----------------|-------|-----------|------------|
| RTX 3090       | 24GB  | ~45-60min | 16         |
| RTX 4090       | 24GB  | ~40-50min | 16         |
| A100 (40GB)    | 40GB  | ~30-45min | 32         |
| A6000          | 48GB  | ~50-70min | 24         |
| RTX 3060 Ti    | 8GB   | ~90-120min| 8          |

### Memory Usage

- **Minimum**: 8GB VRAM (batch_size=4, image_size=320)
- **Recommended**: 16GB+ VRAM (batch_size=16, image_size=320)
- **Optimal**: 24GB+ VRAM (batch_size=32, image_size=416)

## Troubleshooting

### GPU Not Detected

```bash
# Check if CUDA is visible to JAX
uv run python -c "import jax; print(jax.devices())"

# Should show: [CudaDevice(id=0)]
# If showing CpuDevice, reinstall JAX with CUDA:
uv pip uninstall jax jaxlib
uv pip install --upgrade "jax[cuda]>=0.4.20"
```

### Out of Memory Errors

Reduce batch size or image size in `scripts/train.sh`:
```bash
BATCH_SIZE=4  # Reduce from 8
IMAGE_SIZE=256  # Reduce from 320
```

### CUDA Version Mismatch

Lambda Stack should handle CUDA versions automatically. If you encounter issues:
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Ensure JAX CUDA is compatible
uv pip install --upgrade "jax[cuda12]>=0.4.20"  # For CUDA 12.x
# or
uv pip install --upgrade "jax[cuda11]>=0.4.20"  # For CUDA 11.x
```

## Project Structure

```
onnx-yolo-demo-gpu/
├── yolo/                    # Model implementation
│   ├── model.py            # YOLO11n architecture
│   ├── blocks.py           # Building blocks (C3k2, SPPF, etc.)
│   ├── loss.py             # Loss functions
│   └── dataset.py          # Data loading
├── tests/                   # Test suite
│   ├── test_jax_backend.py # JAX/CUDA tests
│   ├── test_model_*.py     # Model tests
│   └── test_onnx_export.py # ONNX export tests
├── scripts/                 # Utility scripts
│   ├── setup.sh            # Setup script
│   ├── train.sh            # Training script
│   └── clean_cache.sh      # Cache cleaning
├── train.py                 # Training entry point
├── export_model.py          # ONNX export script
├── pyproject.toml           # Dependencies (with jax[cuda])
├── QUICKSTART.md            # Quick start guide
└── README.md                # This file
```

## Development

### Running Tests During Development

```bash
# Fast unit tests only
uv run pytest tests/ -v -m "not slow and not browser"

# Include GPU tests
uv run pytest tests/ -v -m "gpu"

# Full test suite
uv run pytest tests/ -v
```

### Cleaning Cache

```bash
bash scripts/clean_cache.sh
```

## Exporting to ONNX

The training script automatically exports to ONNX when training completes. Manual export:

```bash
uv run python export_model.py \
    --checkpoint checkpoints/best_model.npz \
    --output checkpoints/yolo11n_custom.onnx \
    --image-size 320 \
    --num-classes 2
```

## Monitoring

### WandB (Recommended)

```bash
# Login once
wandb login

# Training metrics will be logged to:
# https://wandb.ai/<your-username>/yolo11n-pytensor
```

### Local Logs

```bash
# View training progress
tail -f training.log

# Check checkpoints
ls -lh checkpoints/
```

## Related

- **CPU Version**: [../onnx-yolo-demo](../onnx-yolo-demo) - CPU-only version using `jax[cpu]`
- **PyTensor**: [../../../](../../../) - Main PyTensor repository
- **ONNX**: [../](../) - Other ONNX examples

## License

Same as PyTensor main repository.

## Contributing

This is a demo project. For contributions, please refer to the main PyTensor repository.
