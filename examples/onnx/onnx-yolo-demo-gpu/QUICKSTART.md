# YOLO11n GPU Training - Quick Start (Lambda 22.04 + CUDA)

This version is optimized for Lambda Stack 22.04 with CUDA support. It uses `jax[cuda]` for GPU acceleration.

## Prerequisites

- Lambda Stack 22.04 (Ubuntu 22.04 with CUDA drivers)
- NVIDIA GPU with CUDA support
- Python 3.11+
- Git

## TL;DR - Commands to Run

```bash
# 1. SSH into your Lambda GPU server
ssh user@your-lambda-server

# 2. Clone and navigate to GPU-specific directory
git clone <your-repo-url>
cd pytensor/examples/onnx/onnx-yolo-demo-gpu

# 3. Run setup (one time only)
bash scripts/setup.sh
# This creates a .env file with CUDA-specific settings

# 4. (Optional) Login to WandB for training visualization
wandb login
# Enter API key from https://wandb.ai/authorize
# If skipped, training will run without WandB logging

# 5. Start training in background (includes COCO download)
nohup bash scripts/train.sh > training.log 2>&1 &
# First run will download ~20GB COCO dataset (30-60 min)
# Subsequent runs skip download

# 6. Logout and come back later
exit

# 7. When ready, SSH back and download model
ssh user@your-lambda-server
cd pytensor/examples/onnx/onnx-yolo-demo-gpu

# Check if training is done:
tail training.log

# Download ONNX model to your laptop:
# (Run this from your laptop terminal)
scp user@your-lambda-server:~/pytensor/examples/onnx/onnx-yolo-demo-gpu/checkpoints/yolo11n_best.onnx .
```

## What Gets Created

```
onnx-yolo-demo-gpu/
├── .venv/                         # Python environment (with jax[cuda])
├── data/coco/                     # Dataset (if using real data)
├── checkpoints/
│   ├── best_model.npz            # Best checkpoint
│   ├── yolo11n_best.onnx         # ONNX model for deployment
│   └── checkpoint_epoch_*.npz    # Periodic checkpoints
├── logs/                          # Training logs
└── training.log                   # Training output (when run with nohup)
```

## Monitor Training

### Option 1: WandB Dashboard (Recommended)
Open in your browser:
```
https://wandb.ai/<your-username>/yolo11n-pytensor
```

### Option 2: Check Log File
```bash
ssh user@your-lambda-server
cd pytensor/examples/onnx/onnx-yolo-demo-gpu
tail -f training.log
```

### Option 3: Check if Process is Running
```bash
ps aux | grep train.py
```

## Stop Training (If Needed)

```bash
# Find the process
ps aux | grep train.py

# Kill it (replace PID with actual process ID)
kill <PID>

# A checkpoint will be saved automatically
```

## Resume Training

```bash
uv run python train.py --resume checkpoints/interrupted_checkpoint.npz
```

## Running Tests

To verify the installation and model implementation with CUDA:

```bash
# Run all tests (will use CUDA if available)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model_integration.py -v

# Run GPU-specific tests only
uv run pytest tests/ -v -m gpu

# Run individual test
uv run pytest tests/test_jax_backend.py::test_jax_uses_gpu -v
```

## Typical Timeline (on Lambda GPU)

- **Setup**: 5-10 minutes (one time)
- **COCO Download**: 30-60 minutes (one time, on first train.sh run)
- **Training**:
  - RTX 3090/4090: ~45-60 minutes
  - A100: ~30-45 minutes
  - A6000: ~50-70 minutes
- **Download Model**: < 1 minute (ONNX file is ~10MB)

## After Training

You'll have:
1. **ONNX model**: `checkpoints/yolo11n_best.onnx` - Ready for deployment
2. **Checkpoints**: `checkpoints/*.npz` - Can resume or fine-tune
3. **Training logs**: In WandB dashboard - Loss curves, metrics
4. **Log file**: `training.log` - Full console output

## Common Issues

### "GPU not detected" or "JAX using CPU"
- Check CUDA installation: `nvidia-smi`
- Verify JAX CUDA: `uv run python -c "import jax; print(jax.devices())"`
- Should show: `[CudaDevice(id=0)]`
- If fails, reinstall: `uv pip install --upgrade "jax[cuda]>=0.4.20"`

### "Out of memory"
- Edit `train.sh` and reduce `BATCH_SIZE=4` (or 2)
- Reduce image size: `IMAGE_SIZE=256`

### "WandB not logged in"
- This is OK! Training will continue without WandB logging
- To enable WandB: Stop training, run `wandb login`, then restart
- Or run without logging by adding `--no-wandb` flag manually

### "CUDA version mismatch"
- Lambda Stack should handle CUDA drivers automatically
- If issues persist, check: `nvcc --version` and `nvidia-smi`
- Ensure CUDA 11.8+ or 12.x is installed

## File Sizes

- UV cache: ~500MB
- COCO dataset (optional): ~20GB train, ~1GB val
- Checkpoints: ~20MB each
- ONNX model: ~10MB
- Training log: ~1-5MB

## Need Help?

See `README.md` for detailed documentation.
