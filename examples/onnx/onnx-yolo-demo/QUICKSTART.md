# YOLO11n GPU Server Training - Quick Start

## TL;DR - Commands to Run

```bash
# 1. SSH into your GPU server
ssh user@your-gpu-server

# 2. Clone and navigate to directory
git clone <your-repo-url>
cd pytensor/examples/onnx/onnx-yolo-demo

# 3. Run setup (one time only)
bash scripts/setup.sh
# This creates a .env file with sensible defaults - no need to edit it!

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
ssh user@your-gpu-server
cd pytensor/examples/onnx/onnx-yolo-demo

# Check if training is done:
tail training.log

# Download ONNX model to your laptop:
# (Run this from your laptop terminal)
scp user@your-gpu-server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/yolo11n_best.onnx .
```

## What Gets Created

```
onnx-yolo-demo/
├── venv/                          # Python environment
├── data/coco/                     # Dataset (if using real data)
├── checkpoints/
│   ├── best_model.npz            # Best checkpoint
│   ├── yolo11n_best.onnx         # ONNX model for deployment
│   └── checkpoint_epoch_*.npz    # Periodic checkpoints
└── training.log                   # Training output
```

## Monitor Training

### Option 1: WandB Dashboard (Recommended)
Open in your browser:
```
https://wandb.ai/<your-username>/yolo11n-pytensor
```

### Option 2: Check Log File
```bash
ssh user@your-gpu-server
cd pytensor/examples/onnx/onnx-yolo-demo
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

To verify the installation and model implementation:

```bash
# Run all tests
uv run pytest testing/ -v

# Run specific test file
uv run pytest testing/test_model.py -v

# Run individual test
uv run pytest testing/test_model.py::test_conv_bn_silu -v
```

## Typical Timeline

- **Setup**: 5-10 minutes (one time)
- **COCO Download**: 30-60 minutes (one time, on first train.sh run)
- **Training**: 2-3 hours on Tesla T4, ~30-45 min on A100
- **Download Model**: < 1 minute (ONNX file is ~10MB)

## After Training

You'll have:
1. **ONNX model**: `checkpoints/yolo11n_best.onnx` - Ready for deployment
2. **Checkpoints**: `checkpoints/*.npz` - Can resume or fine-tune
3. **Training logs**: In WandB dashboard - Loss curves, metrics
4. **Log file**: `training.log` - Full console output

## Common Issues

### "GPU not detected"
- Check: `nvidia-smi`
- If fails, contact your GPU server admin

### "Out of memory"
- Edit `train.sh` and reduce `BATCH_SIZE=4` (or 2)

### "WandB not logged in"
- This is OK! Training will continue without WandB logging
- To enable WandB: Stop training, run `wandb login`, then restart
- Or run without logging by adding `--no-wandb` flag manually

## File Sizes

- UV cache: ~500MB
- COCO dataset (optional): ~20GB train, ~1GB val
- Checkpoints: ~20MB each
- ONNX model: ~10MB
- Training log: ~1-5MB

## Need Help?

See `README.md` for detailed documentation.
