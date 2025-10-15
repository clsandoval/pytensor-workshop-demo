# Environment Variables Setup Guide

## Quick Start - Do You Need to Set Anything?

**Short answer: NO** - The `setup.sh` script creates a `.env` file with sensible defaults that will work for most GPU servers.

**You only need to configure environment variables if:**
- You have multiple GPUs and want to use specific ones
- You want to limit GPU memory usage
- You want to set your WandB API key in the environment (instead of `wandb login`)
- You're experiencing GPU memory issues

## What `setup.sh` Creates Automatically

When you run `bash setup.sh`, it creates a `.env` file with these defaults:

```bash
# PyTensor Configuration
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"

# JAX GPU Memory Configuration
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# WandB Configuration
WANDB_PROJECT=yolo11n-pytensor

# System Configuration
PYTHONUNBUFFERED=1
```

**This is all you need for basic usage!**

## When to Customize Environment Variables

### Scenario 1: Multiple GPUs - Use Specific GPU

If your server has multiple GPUs and you want to use only GPU 0:

```bash
# Edit .env file:
echo "CUDA_VISIBLE_DEVICES=0" >> .env
```

Or to use GPUs 1 and 2:
```bash
echo "CUDA_VISIBLE_DEVICES=1,2" >> .env
```

### Scenario 2: GPU Out of Memory Errors

If you get "Out of Memory" errors, reduce the memory fraction:

```bash
# Edit .env file, change:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Use only 70% of GPU memory
```

Or disable memory preallocation:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Scenario 3: Set WandB API Key (Alternative to `wandb login`)

Instead of running `wandb login`, you can set the API key in `.env`:

```bash
# Get your key from: https://wandb.ai/authorize
echo "WANDB_API_KEY=your_key_here" >> .env
```

### Scenario 4: Disable WandB Completely

If you don't want to use WandB at all:

```bash
echo "WANDB_DISABLED=true" >> .env
```

## All Available Environment Variables

See `.env.example` for a complete list of all available configuration options.

```bash
# Copy example to see all options:
cp .env.example .env.custom
nano .env.custom  # Edit as needed
mv .env.custom .env
```

## Common Environment Variable Reference

| Variable | Purpose | Default | When to Change |
|----------|---------|---------|----------------|
| `PYTENSOR_FLAGS` | PyTensor backend config | `device=cuda,...` | Rarely needed |
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to use | All GPUs | Multiple GPU systems |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | GPU memory limit | `0.9` (90%) | Out of memory errors |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Pre-allocate memory | `true` | Memory conflicts |
| `WANDB_API_KEY` | WandB authentication | Not set | Alternative to `wandb login` |
| `WANDB_PROJECT` | WandB project name | `yolo11n-pytensor` | Personal preference |
| `WANDB_DISABLED` | Disable WandB | Not set | Don't want logging |

## How Environment Variables Are Loaded

The `train.sh` script automatically loads `.env`:

```bash
# This happens automatically when you run train.sh:
source .env  # Loads all variables
```

## Verification

To check if environment variables are loaded correctly:

```bash
# After running setup.sh:
source venv/bin/activate
source .env

# Check PyTensor config:
echo $PYTENSOR_FLAGS

# Check CUDA devices:
echo $CUDA_VISIBLE_DEVICES

# Check WandB project:
echo $WANDB_PROJECT

# Verify GPU is detected:
python -c "import pytensor; print(pytensor.config.device)"

# Verify JAX sees GPU:
python -c "import jax; print(jax.devices())"
```

## Troubleshooting

### "PyTensor not using GPU"

Check your PYTENSOR_FLAGS:
```bash
echo $PYTENSOR_FLAGS
# Should show: device=cuda,floatX=float32,optimizer=fast_run
```

If empty, reload .env:
```bash
source .env
```

### "JAX not detecting GPU"

Check CUDA devices:
```bash
nvidia-smi  # Should show your GPU
echo $CUDA_VISIBLE_DEVICES  # Should show GPU numbers or be empty (all GPUs)
```

Test JAX:
```bash
python -c "
import jax
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
"
```

### "Out of Memory" Even with Small Batch Size

Lower the memory fraction in `.env`:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5  # Use only 50%
```

Or disable preallocation:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## What You DON'T Need to Set

You **don't** need to set:
- `LD_LIBRARY_PATH` - Usually auto-detected by CUDA
- `JAX_PLATFORM_NAME` - Auto-detected
- `EPOCHS`, `BATCH_SIZE`, etc. - Use command-line args instead
- Most PyTensor compilation settings - Defaults are good

## Example Configurations

### Minimal (Default) - Good for Most Cases
```bash
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
PYTHONUNBUFFERED=1
```

### Multi-GPU Server - Use GPU 1 Only
```bash
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"
CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
PYTHONUNBUFFERED=1
```

### Low Memory GPU (< 8GB)
```bash
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
PYTHONUNBUFFERED=1
```

### No WandB Logging
```bash
PYTENSOR_FLAGS="device=cuda,floatX=float32,optimizer=fast_run"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
WANDB_DISABLED=true
PYTHONUNBUFFERED=1
```

## Summary

**For 95% of users**: Just run `bash setup.sh` and you're done. The defaults work great!

**If you have specific needs**: Edit `.env` based on the scenarios above, or copy `.env.example` for all options.

**To verify everything works**: Run the verification commands in the "Verification" section above.
