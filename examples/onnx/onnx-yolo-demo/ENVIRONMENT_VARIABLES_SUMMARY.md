# Environment Variables - Quick Answer

## Do I need to set any environment variables?

**NO!** The `setup.sh` script does it for you automatically.

## What `setup.sh` does:

When you run `bash setup.sh`, it automatically creates a `.env` file with:

```bash
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
WANDB_PROJECT=yolo11n-pytensor
PYTHONUNBUFFERED=1
```

This configuration:
- ✅ Enables GPU training with PyTensor
- ✅ Optimizes JAX GPU memory usage (90% of available memory)
- ✅ Sets up WandB project name
- ✅ Enables unbuffered Python output for better logging

## Your workflow (no manual env var setup needed):

```bash
# 1. Setup (creates .env automatically)
bash setup.sh

# 2. Train (loads .env automatically)
bash train.sh
```

That's it!

## When you MIGHT need to edit .env:

### Multiple GPUs? Choose which one to use:
```bash
# Edit .env, add this line:
CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

### Out of memory? Reduce GPU memory usage:
```bash
# Edit .env, change this line:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Use 70% instead of 90%
```

### Don't want WandB? Disable it:
```bash
# Edit .env, add this line:
WANDB_DISABLED=true
```

## How to edit .env if needed:

```bash
# SSH into your server
cd pytensor/examples/onnx/onnx-yolo-demo

# Edit the file
nano .env  # or vim .env

# Save and run training
bash train.sh
```

## See all options:

Check `.env.example` for a complete list of all available settings:
```bash
cat .env.example
```

Or read the full guide:
```bash
cat ENV_SETUP.md
```

## Summary

**99% of users**: Don't touch anything. `setup.sh` creates perfect defaults.

**Advanced users**: Edit `.env` only if you have specific needs (multiple GPUs, memory limits, etc.)

**The `.env` file is automatically loaded** every time you run `train.sh`, so you never need to manually export environment variables.
