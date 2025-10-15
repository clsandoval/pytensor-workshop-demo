# YOLO11n Training on GPU Server

Complete training setup for YOLO11n object detection model using PyTensor with JAX backend.

## Quick Start (SSH to GPU Server)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd pytensor/examples/onnx/onnx-yolo-demo

# 2. Run setup (one time only)
bash setup.sh

# 3. Login to WandB (for training visualization)
source venv/bin/activate
wandb login
# Enter your API key from https://wandb.ai/authorize

# 4. Start training
bash train.sh

# 5. To run in background and logout:
nohup bash train.sh > training.log 2>&1 &

# Monitor training:
tail -f training.log

# Or check WandB dashboard:
# https://wandb.ai/<your-username>/yolo11n-pytensor
```

## Downloading the Trained Model

After training completes, download the ONNX model to your laptop:

```bash
# From your laptop:
scp user@gpu-server:~/path/to/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/yolo11n_best.onnx .
```

## Project Structure

```
onnx-yolo-demo/
├── setup.sh              # Environment setup script
├── train.sh              # Training wrapper script
├── train.py              # Main training script
├── model.py              # YOLO11n architecture
├── blocks.py             # Model building blocks
├── loss.py               # Detection loss functions
├── dataset.py            # COCO dataset loader
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── data/                # Dataset directory (created by setup)
│   └── coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
│
├── checkpoints/         # Model checkpoints (created during training)
│   ├── best_model.npz
│   ├── yolo11n_best.onnx
│   └── checkpoint_epoch_*.npz
│
└── venv/               # Virtual environment (created by setup)
```

## What Each Script Does

### `setup.sh` - One-Time Environment Setup

This script:
- Checks for GPU availability
- Creates Python virtual environment
- Installs PyTensor from source
- Installs JAX with CUDA support
- Installs training dependencies (WandB, COCO tools, etc.)
- Creates necessary directories

**Run once after cloning the repo.**

### `train.sh` - Training Wrapper

This script:
- Activates the virtual environment
- Checks WandB login status
- Detects GPU and adjusts batch size automatically
- Runs training with sensible defaults
- Exports model to ONNX after training

**Run this to start training.**

### `train.py` - Main Training Script

Full-featured training script with:
- WandB experiment tracking
- Checkpoint saving and resuming
- SGD optimizer with momentum
- Synthetic data (for testing) or COCO dataset support
- ONNX export after training

## Training Configuration

Default settings in `train.sh`:
- **Epochs**: 100
- **Batch Size**: 8 (auto-adjusted based on GPU memory)
- **Learning Rate**: 0.01
- **Image Size**: 320x320
- **Classes**: 2 (person, cellphone)

To customize, edit `train.sh` or run `train.py` directly:

```bash
python train.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.02 \
    --image-size 320 \
    --num-classes 2
```

## Using Real COCO Dataset

The current setup uses **synthetic random data** for demonstration and testing the pipeline.

To train on real COCO data:

1. **Download COCO dataset** (done automatically on first run):
```bash
python dataset.py --data-dir ./data/coco --split train
```

2. **Modify `train.py`** to use real dataloader:
   - Replace `SyntheticDataloader` with `COCODataset` (line ~350)
   - The code structure is already in place

## WandB Integration

The training script automatically logs to [Weights & Biases](https://wandb.ai):

- **Loss curves**: Total, box, and classification losses
- **Learning rate**: Current learning rate
- **Checkpoints**: Automatically saved
- **System metrics**: GPU usage, memory, etc.

### First Time WandB Setup

```bash
# After running setup.sh:
source venv/bin/activate
wandb login

# Enter your API key from: https://wandb.ai/authorize
```

### Disable WandB

To train without WandB logging:

```bash
python train.py --no-wandb
```

## Monitoring Training

### Option 1: WandB Dashboard (Recommended)

Visit: `https://wandb.ai/<your-username>/yolo11n-pytensor`

Real-time metrics:
- Loss curves
- Training speed (batches/sec)
- GPU utilization
- System metrics

### Option 2: Log File

```bash
# If running in background:
tail -f training.log
```

### Option 3: SSH and Check Directly

```bash
# In another SSH session:
source venv/bin/activate
python -c "
import numpy as np
ckpt = np.load('checkpoints/best_model.npz')
print(f'Best loss: {ckpt[\"best_loss\"]:.6f}')
print(f'Epoch: {ckpt[\"epoch\"]}')
"
```

## Checkpointing

The training script automatically saves checkpoints:

- **Every 10 epochs**: `checkpoint_epoch_10.npz`, `checkpoint_epoch_20.npz`, etc.
- **Best model**: `best_model.npz` (lowest loss)
- **On interrupt**: `interrupted_checkpoint.npz` (if you press Ctrl+C)
- **On error**: `error_checkpoint.npz` (if training crashes)

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_50.npz
```

## GPU Server Workflow

### Initial Setup (Once)

```bash
# SSH into GPU server
ssh user@gpu-server

# Clone repo
git clone <your-repo-url>
cd pytensor/examples/onnx/onnx-yolo-demo

# Run setup
bash setup.sh

# Login to WandB
source venv/bin/activate
wandb login
```

### Start Training (Background)

```bash
# Start training in background
nohup bash train.sh > training.log 2>&1 &

# Get the process ID
echo $! > training.pid

# You can now logout and training will continue
```

### Check Training Progress

```bash
# SSH back in anytime
ssh user@gpu-server
cd pytensor/examples/onnx/onnx-yolo-demo

# Check log
tail -f training.log

# Or check WandB dashboard from your laptop browser
```

### Download Trained Model

```bash
# From your laptop:
scp user@gpu-server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/yolo11n_best.onnx .

# Verify the file:
ls -lh yolo11n_best.onnx
```

### Stop Training (If Needed)

```bash
# Find the process
cat training.pid  # or: ps aux | grep train.py

# Kill the process
kill $(cat training.pid)

# The script will save a checkpoint before exiting
```

## Model Details

**YOLO11n (Nano)**:
- **Parameters**: ~2.3 million
- **Input**: 320x320 RGB images
- **Output**: 3 detection scales (P3, P4, P5)
- **Classes**: 2 (person, cellphone) - configurable
- **Architecture**:
  - Backbone: CSP bottleneck with C3k2 blocks
  - Neck: FPN-PAN with SPPF
  - Head: Anchor-free detection

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install NVIDIA drivers:
# (varies by system - consult your GPU server documentation)
```

### CUDA Version Mismatch

Edit `setup.sh` and change JAX installation line:

```bash
# For CUDA 11:
pip install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12:
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Out of Memory

Reduce batch size in `train.sh`:

```bash
BATCH_SIZE=4  # or even 2 for GPUs with < 8GB memory
```

### Slow Training

Ensure:
1. GPU is detected (`nvidia-smi`)
2. JAX is using GPU:
   ```python
   import jax
   print(jax.devices())  # Should show GPU
   ```
3. PyTensor is configured for GPU:
   ```bash
   export PYTENSOR_FLAGS="device=cuda,floatX=float32"
   ```

### WandB Login Issues

```bash
# Re-login:
wandb login --relogin

# Or train without WandB:
python train.py --no-wandb
```

## Next Steps After Training

1. **Download the ONNX model** to your laptop (see above)

2. **Test in browser** using ONNX Runtime Web:
   - See `webgpu_vs_wasm_benchmark.html` for example
   - Load your trained model: `yolo11n_best.onnx`
   - Run inference on webcam or images

3. **Evaluate on test set**:
   - Use `dataset.py` to load validation data
   - Compute mAP (mean Average Precision)
   - Compare against baseline

4. **Fine-tune** on your own data:
   - Replace COCO dataset with custom dataset
   - Adjust `dataset.py` for your annotation format
   - Re-run training

## Performance Expectations

**Training Time** (approximate):
- **GPU**: Tesla T4: ~2-3 hours for 100 epochs
- **GPU**: A100: ~30-45 minutes for 100 epochs
- **CPU**: Not recommended (would take days)

**Model Performance**:
- This is a **demo/proof-of-concept** implementation
- Loss function is simplified (basic regularization)
- For production use, implement proper:
  - Target assignment
  - IoU-based losses (CIoU/GIoU)
  - Balanced loss weights
  - Data augmentation

## References

- **PyTensor**: https://github.com/pymc-devs/pytensor
- **YOLO11**: https://github.com/ultralytics/ultralytics
- **WandB**: https://wandb.ai/
- **COCO Dataset**: https://cocodataset.org/

## Support

For issues:
1. Check `training.log` for error messages
2. Verify `setup.sh` completed successfully
3. Ensure GPU is detected and CUDA is working
4. Check WandB dashboard for training metrics

## License

See main PyTensor repository for license information.
