# SSH Training Guide - YOLO11n on GPU Server

Complete walkthrough for SSHing into your GPU server and training your YOLO11n model with PyTensor.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
- [Monitoring Training](#monitoring-training)
- [Downloading Your Model](#downloading-your-model)
- [Troubleshooting](#troubleshooting)
- [Advanced Operations](#advanced-operations)

---

## Prerequisites

### On Your Local Machine
- SSH client (built into macOS/Linux/Windows 10+)
- Git (if cloning from your machine)
- SCP for downloading models

### On Your GPU Server
- NVIDIA GPU with CUDA support
- Ubuntu/Linux OS
- Python 3.8+
- Git installed

### Verify GPU Access
First time SSHing? Check GPU availability:
```bash
ssh user@your-gpu-server
nvidia-smi
```

You should see your GPU listed (e.g., Tesla T4, A100, etc.).

---

## Step-by-Step Setup

### 1. SSH into Your GPU Server
```bash
ssh user@your-gpu-server
```

**Replace:**
- `user` with your username
- `your-gpu-server` with the server hostname or IP address

**Example:**
```bash
ssh john@192.168.1.100
# or
ssh john@ml-server.example.com
```

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd pytensor/examples/onnx/onnx-yolo-demo
```

**If you already cloned it:**
```bash
cd pytensor/examples/onnx/onnx-yolo-demo
git pull origin main  # Get latest changes
```

### 3. Run One-Time Setup
```bash
bash setup.sh
```

**What this does:**
- Creates a Python virtual environment (`venv/`)
- Installs all dependencies (PyTensor, JAX, etc.)
- Creates a `.env` file with sensible defaults
- No manual configuration needed!

**Expected output:**
```
Creating virtual environment...
Installing dependencies...
Setup complete!
```

**Time:** ~5-10 minutes

### 4. (Optional) Configure WandB Logging
WandB provides beautiful training dashboards with loss curves and metrics.

```bash
source venv/bin/activate
wandb login
```

Enter your API key from: https://wandb.ai/authorize

**Skip this if:**
- You don't want training visualization
- You're doing a quick test run
- Training will still work without WandB

### 5. Start Training in Background
```bash
nohup bash train.sh > training.log 2>&1 &
```

**What this does:**
- `nohup` - Keeps training running after you logout
- `train.sh` - Main training script
- `> training.log` - Saves all output to a file
- `2>&1` - Includes errors in the log
- `&` - Runs in background

**First Run Warning:**
The first time you run this, it will download the COCO dataset (~20GB). This takes 30-60 minutes depending on your connection.

**Subsequent runs skip the download automatically.**

### 6. Check Training Started
```bash
ps aux | grep train.py
```

You should see a running Python process.

**Alternative check:**
```bash
tail training.log
```

Should show training initialization messages.

### 7. Logout Safely
```bash
exit
```

Your training continues in the background!

---

## Monitoring Training

### Option 1: WandB Dashboard (Recommended)

If you logged into WandB, open in your browser:
```
https://wandb.ai/<your-username>/yolo11n-pytensor
```

**You can see:**
- Real-time loss curves
- Validation metrics
- GPU utilization
- Training speed (iterations/sec)
- Estimated time remaining

### Option 2: SSH Back and Check Logs

```bash
ssh user@your-gpu-server
cd pytensor/examples/onnx/onnx-yolo-demo

# View last 20 lines
tail -n 20 training.log

# Follow in real-time (Ctrl+C to stop)
tail -f training.log

# Search for specific info
grep "Epoch" training.log
grep "loss" training.log
```

### Option 3: Check Process Status

```bash
ssh user@your-gpu-server

# Check if still running
ps aux | grep train.py

# Check GPU usage
nvidia-smi

# Watch GPU usage in real-time (updates every 2 sec)
watch -n 2 nvidia-smi
```

### What to Look For

**Good signs:**
```
Epoch 1/50 - Loss: 2.345
Epoch 2/50 - Loss: 2.123
Validation mAP: 0.234
```

**Warning signs:**
```
CUDA out of memory  # Need to reduce batch size
Loss: NaN           # Training instability
No progress for 10+ min  # Check logs
```

---

## Downloading Your Model

### Check Training Completion

```bash
ssh user@your-gpu-server
cd pytensor/examples/onnx/onnx-yolo-demo

# Check last lines of log
tail training.log

# Look for:
# "Training complete!"
# "Best model saved to checkpoints/yolo11n_best.onnx"
```

### Download ONNX Model

**From your local machine** (not the server):

```bash
scp user@your-gpu-server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/yolo11n_best.onnx .
```

**Downloads to your current directory** (~10MB file)

### Download All Checkpoints (Optional)

```bash
scp -r user@your-gpu-server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/ ./checkpoints/
```

### Verify Downloaded Model

```bash
ls -lh yolo11n_best.onnx
# Should show ~10MB file
```

---

## Troubleshooting

### Issue: "GPU not detected"

**Check:**
```bash
nvidia-smi
```

**If command fails:**
- Contact your server administrator
- NVIDIA drivers may not be installed
- You might not have GPU access permissions

### Issue: "Out of memory"

**Solution:** Reduce batch size

```bash
# Edit train.sh
nano train.sh

# Change this line:
export BATCH_SIZE=8

# To smaller value:
export BATCH_SIZE=4  # or even 2
```

Save (Ctrl+O, Enter) and exit (Ctrl+X), then restart training.

### Issue: "Training seems stuck"

**Check:**
```bash
tail -n 50 training.log
nvidia-smi
htop  # Check CPU/RAM usage
```

**Common causes:**
- Downloading dataset (first run only)
- Checkpoint saving (happens every few epochs)
- Validation running (slower than training)

### Issue: "WandB not logged in"

**This is OK!** Training will continue without WandB logging.

**To enable WandB:**
1. Stop training: `kill <PID>`
2. Login: `wandb login`
3. Restart: `nohup bash train.sh > training.log 2>&1 &`

### Issue: "Can't SSH - connection refused"

**Check:**
- Is the server on? Ask administrator
- Correct hostname/IP? Double-check
- VPN required? Connect first
- Firewall blocking? Contact IT

### Issue: "Training stopped unexpectedly"

**Check:**
```bash
tail -n 100 training.log
# Look for error messages at the end
```

**Common causes:**
- Out of memory (reduce batch size)
- Out of disk space (clean up old files)
- Server reboot (contact admin)

**Resume training:**
```bash
source venv/bin/activate
python train.py --resume checkpoints/checkpoint_epoch_*.npz
```

---

## Advanced Operations

### Stop Training

```bash
# Find process ID
ps aux | grep train.py

# Stop it (replace 12345 with actual PID)
kill 12345
```

A checkpoint is automatically saved before stopping.

### Resume from Checkpoint

```bash
source venv/bin/activate

# Resume from specific checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.npz

# Resume from interrupted training
python train.py --resume checkpoints/interrupted_checkpoint.npz
```

### Adjust Training Parameters

Edit `train.sh` before starting:
```bash
nano train.sh
```

**Common parameters:**
```bash
export EPOCHS=50          # Number of training epochs
export BATCH_SIZE=8       # Samples per batch
export LEARNING_RATE=0.001  # Learning rate
export USE_COCO=true      # Use real COCO data vs synthetic
```

### Run on Specific GPU

If you have multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash train.sh > training.log 2>&1 &
```

Replace `0` with GPU ID from `nvidia-smi`.

### Transfer Large Files

**Use rsync for reliability:**
```bash
rsync -avz --progress user@server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/ ./checkpoints/
```

Benefits over scp:
- Shows progress bar
- Resumes if interrupted
- Only transfers new/changed files

### Background Session Management

**Use tmux or screen for better session management:**

```bash
# Install tmux (one time)
sudo apt install tmux

# Start tmux session
tmux new -s training

# Run training
bash train.sh

# Detach (training continues): Ctrl+B then D

# Logout safely
exit

# Later, reattach
ssh user@server
tmux attach -t training
```

---

## Timeline Expectations

| Task | Duration | Notes |
|------|----------|-------|
| Setup (first time) | 5-10 min | One time only |
| COCO download | 30-60 min | First run only |
| Training (Tesla T4) | 2-3 hours | 50 epochs |
| Training (A100) | 30-45 min | 50 epochs |
| Model download | < 1 min | 10MB file |

---

## Files Created

After training completes:

```
onnx-yolo-demo/
├── venv/                          # Python environment (~2GB)
├── data/coco/                     # Dataset (~20GB, if using COCO)
│   ├── train2017/
│   └── val2017/
├── checkpoints/                   # Model checkpoints
│   ├── best_model.npz            # Best checkpoint (~20MB)
│   ├── yolo11n_best.onnx         # ONNX export (~10MB) ← DEPLOY THIS
│   ├── checkpoint_epoch_10.npz
│   ├── checkpoint_epoch_20.npz
│   └── interrupted_checkpoint.npz
└── training.log                   # Full training output (1-5MB)
```

---

## Quick Reference Commands

```bash
# SSH in
ssh user@server

# Navigate to project
cd pytensor/examples/onnx/onnx-yolo-demo

# Check if running
ps aux | grep train.py

# View logs
tail -f training.log

# Check GPU
nvidia-smi

# Download model (from local machine)
scp user@server:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/yolo11n_best.onnx .

# Stop training
kill $(pgrep -f train.py)
```

---

## Getting Help

- **Full documentation:** See `README.md` in this directory
- **Environment variables:** See `ENVIRONMENT_VARIABLES_SUMMARY.md`
- **Setup issues:** See `ENV_SETUP.md`
- **GPU training details:** See `thoughts/shared/research/2025-10-15_07-28-53_gpu-training-support.md`

---

## Next Steps After Training

Once you have `yolo11n_best.onnx`:

1. **Test locally** with ONNX Runtime
2. **Deploy to web** using ONNX.js or WebGPU
3. **Deploy to mobile** using ONNX Mobile
4. **Deploy to edge devices** using TensorRT or OpenVINO
5. **Fine-tune** on your own dataset

See deployment guides in `examples/onnx/markdown/` for more details.
