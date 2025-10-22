# ONNX YOLO Demo GPU-Specific Version for Lambda Stack 22.04

## Overview

Create a GPU-specific copy of the YOLO11n demo optimized for Lambda Stack Ubuntu 22.04 environments, configured to train exclusively on JAX CUDA backend. This version will be placed at the root level (`onnx-yolo-demo-gpu/`) rather than under `examples/onnx/` to emphasize its standalone nature as a Lambda Stack deployment.

## Current State Analysis

The existing `examples/onnx/onnx-yolo-demo/` project has:
- **JAX backend**: Currently uses `jax[cpu]` in pyproject.toml:12
- **GPU detection logic**: setup.sh:22-29 and train.py:48-75 check for GPU but fall back to CPU
- **Environment configuration**: .env.example includes JAX_PLATFORMS="cuda" but project dependencies use CPU-only JAX
- **Setup script**: Generic Linux setup with optional GPU support
- **Training script**: Gracefully degrades to CPU if GPU unavailable

### Key Files:
- `pyproject.toml` - Project dependencies (uses jax[cpu])
- `scripts/setup.sh` - Environment setup script
- `scripts/train.sh` - Training execution script
- `.env.example` - Environment variable template
- `train.py` - Main training logic with GPU detection
- `yolo/` - Model implementation (4 Python files)
- `tests/` - Test suite (10 files)

## Desired End State

A new directory `onnx-yolo-demo-gpu/` at the root level containing:
- Modified `pyproject.toml` with `jax[cuda12]` dependency
- Lambda Stack 22.04-specific setup script that REQUIRES CUDA
- Simplified training script that fails fast without GPU
- Updated documentation emphasizing Lambda Stack requirements
- All GPU-required environment variables set by default
- Same model code and tests (unchanged)

### Success Criteria:

#### Automated Verification:
- [ ] Directory structure copied: `ls onnx-yolo-demo-gpu/` shows all files
- [ ] pyproject.toml updated: `grep "jax\[cuda12\]" onnx-yolo-demo-gpu/pyproject.toml` succeeds
- [ ] Setup script requires CUDA: `grep "exit 1" onnx-yolo-demo-gpu/scripts/setup.sh` shows GPU requirement
- [ ] Python imports succeed: `cd onnx-yolo-demo-gpu && uv run python -c "import yolo.model"`
- [ ] Tests can be discovered: `cd onnx-yolo-demo-gpu && uv run pytest --collect-only tests/`

#### Manual Verification:
- [ ] Setup script fails gracefully on non-GPU system
- [ ] Setup script succeeds on Lambda Stack 22.04 with CUDA
- [ ] Training script immediately detects and uses GPU
- [ ] JAX forces CUDA platform (no CPU fallback)
- [ ] Documentation clearly states Lambda Stack 22.04 requirement

## What We're NOT Doing

- NOT modifying the original `examples/onnx/onnx-yolo-demo/` (keep it as cross-platform reference)
- NOT adding new model features or changing model architecture
- NOT creating multi-GPU or distributed training support
- NOT supporting Lambda Stack versions other than 22.04
- NOT adding automatic Lambda Stack installation/configuration
- NOT modifying test logic (only test configuration if needed)

## Implementation Approach

1. **Copy directory structure** to root level as `onnx-yolo-demo-gpu/`
2. **Update dependencies** to enforce CUDA-only JAX
3. **Harden setup script** to fail without NVIDIA GPU
4. **Simplify training logic** to remove CPU fallback paths
5. **Update documentation** to emphasize Lambda Stack requirements
6. **Verify GPU enforcement** through configuration and error messages

## Phase 1: Directory Structure and Dependency Updates

### Overview
Copy the entire demo to root level and update Python dependencies to require JAX with CUDA support.

### Changes Required:

#### 1. Copy Project Directory
**Action**: Copy entire directory to root level

```bash
# From pytensor root
cp -r examples/onnx/onnx-yolo-demo/ onnx-yolo-demo-gpu/
```

#### 2. Update pyproject.toml
**File**: `onnx-yolo-demo-gpu/pyproject.toml`
**Changes**:
- Line 2: Update name to `"onnx-yolo-demo-gpu"`
- Line 4: Update description to include "Lambda Stack 22.04 GPU-only"
- Line 11: Change `"jax[cpu]>=0.4.20"` to `"jax[cuda12]>=0.4.35"`
- Add requirement comment explaining Lambda Stack compatibility

```toml
[project]
name = "onnx-yolo-demo-gpu"
version = "0.1.0"
description = "YOLO11n Object Detection Demo with PyTensor, JAX CUDA, and ONNX (Lambda Stack 22.04)"
requires-python = ">=3.11"
dependencies = [
    # NOTE: pytensor should be installed from root repo in editable mode FIRST:
    #   cd /path/to/pytensor && uv pip install -e ".[development,onnx]"
    # We do NOT include pytensor here to avoid overriding the local dev version
    "numpy>=2.0",
    # JAX with CUDA 12 support for Lambda Stack 22.04
    # Lambda Stack provides CUDA 12.x drivers (requires >= 525.x)
    "jax[cuda12]>=0.4.35",
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "pytest>=7.0",
    "pytest-cov>=4.0.0",
    "pytest-benchmark>=4.0.0",
    "hypothesis>=6.100.0",
]
```

### Success Criteria:

#### Automated Verification:
- [ ] Directory exists: `test -d onnx-yolo-demo-gpu && echo "✓ Directory created"`
- [ ] All files copied: `diff -r examples/onnx/onnx-yolo-demo/ onnx-yolo-demo-gpu/ --exclude=.venv --exclude=__pycache__` shows only expected changes
- [ ] pyproject.toml updated: `grep 'jax\[cuda12\]' onnx-yolo-demo-gpu/pyproject.toml`
- [ ] Project name updated: `grep 'name = "onnx-yolo-demo-gpu"' onnx-yolo-demo-gpu/pyproject.toml`

#### Manual Verification:
- [ ] pyproject.toml is valid TOML syntax
- [ ] All source files are present in new directory

---

## Phase 2: Setup Script Lambda Stack Hardening

### Overview
Modify setup.sh to explicitly require Lambda Stack 22.04 and fail fast without NVIDIA GPU.

### Changes Required:

#### 1. Update setup.sh Header
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Lines 1-6

```bash
#!/bin/bash
# YOLO11n GPU Training Environment Setup Script
# REQUIRES: Lambda Stack 22.04 with NVIDIA GPU
# Run this once after provisioning your Lambda GPU instance
#
# Usage: bash setup.sh
```

#### 2. Add Lambda Stack Detection
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: After line 20, add Lambda Stack version check

```bash
# Check for Lambda Stack (optional - warn if not present)
echo "[1/8] Checking environment..."
if [ -f "/etc/os-release" ]; then
    OS_VERSION=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
    if [ "$OS_VERSION" = "22.04" ]; then
        echo "✓ Ubuntu 22.04 detected"
    else
        echo "⚠ Warning: This script is optimized for Ubuntu 22.04 (Lambda Stack)"
        echo "  Current version: $OS_VERSION"
    fi
fi
echo ""
```

#### 3. Enforce GPU Requirement
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Replace lines 22-30 (GPU check section)

```bash
# Check for GPU (REQUIRED - fail if not present)
echo "[2/8] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    # Check minimum driver version for CUDA 12
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -lt 525 ]; then
        echo "❌ ERROR: NVIDIA driver version $DRIVER_VERSION is too old for CUDA 12"
        echo "   Required: >= 525.x (Lambda Stack provides this automatically)"
        echo "   Please update your driver or use Lambda Stack"
        exit 1
    fi
    echo "✓ Driver version $DRIVER_VERSION is compatible with CUDA 12"
else
    echo "❌ ERROR: No NVIDIA GPU detected!"
    echo "   This version requires GPU for training."
    echo "   Please run on a Lambda Stack GPU instance or use the CPU version:"
    echo "     examples/onnx/onnx-yolo-demo/"
    exit 1
fi
echo ""
```

#### 4. Update Section Numbers
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Update all remaining section numbers from [3/7] → [3/8], [4/7] → [4/8], etc.

#### 5. Update PyTensor Installation Path
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Line 87 - Update path since we're now at root level

```bash
# CRITICAL: Install PyTensor from parent directory in EDITABLE mode FIRST
# This ensures the local development version is used, not PyPI
echo "Installing PyTensor from source (editable)..."
cd ../  # Go up one level from onnx-yolo-demo-gpu/ to pytensor/
uv pip install -e ".[development,onnx]"
```

#### 6. Return to Correct Directory
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Line 102 - Update return path

```bash
# Return to demo directory
cd onnx-yolo-demo-gpu/
```

#### 7. Update Final Instructions
**File**: `onnx-yolo-demo-gpu/scripts/setup.sh`
**Changes**: Lines 160-185 - Update instructions to reference Lambda Stack

```bash
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
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
echo "   scp user@lambda:$(pwd)/checkpoints/yolo11n_best.onnx ."
echo ""
echo "5. To run tests (GPU required):"
echo "   uv run pytest tests/ -v -m gpu"
echo "   # Or all tests: uv run pytest tests/ -v"
echo ""
echo "=========================================="
```

### Success Criteria:

#### Automated Verification:
- [ ] Script has correct shebang: `head -1 onnx-yolo-demo-gpu/scripts/setup.sh | grep "#!/bin/bash"`
- [ ] GPU check enforces exit: `grep -A5 "No NVIDIA GPU detected" onnx-yolo-demo-gpu/scripts/setup.sh | grep "exit 1"`
- [ ] Driver version check present: `grep "DRIVER_MAJOR" onnx-yolo-demo-gpu/scripts/setup.sh`
- [ ] Correct path navigation: `grep "cd ../" onnx-yolo-demo-gpu/scripts/setup.sh`

#### Manual Verification:
- [ ] Script fails immediately on system without nvidia-smi
- [ ] Script provides helpful error message pointing to CPU version
- [ ] Script checks for minimum NVIDIA driver version 525

---

## Phase 3: Training Script GPU Enforcement

### Overview
Simplify train.sh and train.py to fail fast without GPU, removing all CPU fallback logic.

### Changes Required:

#### 1. Update train.sh Header
**File**: `onnx-yolo-demo-gpu/scripts/train.sh`
**Changes**: Lines 1-5

```bash
#!/bin/bash
# YOLO11n GPU Training Script
# REQUIRES: Lambda Stack 22.04 with NVIDIA GPU
#
# Usage: bash train.sh
```

#### 2. Enforce GPU Requirement in train.sh
**File**: `onnx-yolo-demo-gpu/scripts/train.sh`
**Changes**: Lines 45-56 - Make GPU check fatal

```bash
# Check for GPU (REQUIRED)
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
else
    echo "❌ ERROR: No GPU detected!"
    echo "   This version requires NVIDIA GPU for training."
    echo "   Please use Lambda Stack or switch to CPU version: examples/onnx/onnx-yolo-demo/"
    exit 1
fi
```

#### 3. Enforce CUDA Platform
**File**: `onnx-yolo-demo-gpu/scripts/train.sh`
**Changes**: Lines 115-122 - Remove conditional CPU fallback

```bash
# CRITICAL: Force JAX to use CUDA (no CPU fallback)
export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
export JAX_PLATFORMS="cuda"
export JAX_ENABLE_X64=False
echo "PyTensor config: $PYTENSOR_FLAGS"
echo "JAX platforms: $JAX_PLATFORMS (CUDA ONLY - will fail if GPU unavailable)"
echo ""
```

#### 4. Update train.py GPU Detection
**File**: `onnx-yolo-demo-gpu/train.py`
**Changes**: Lines 47-76 - Make GPU required, remove CPU fallback

```python
# Configure PyTensor to use JAX backend for GPU acceleration
try:
    import jax

    # Force JAX to use GPU
    devices = jax.devices()

    if len(devices) == 0:
        raise RuntimeError("No JAX devices available!")

    device_type = devices[0].platform

    if device_type != "gpu":
        raise RuntimeError(
            f"ERROR: JAX device platform is '{device_type}' but must be 'gpu'!\n"
            f"This GPU-specific version requires CUDA.\n"
            f"Available devices: {devices}\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Ensure you're running on Lambda Stack 22.04 with NVIDIA GPU\n"
            f"  2. Check nvidia-smi shows available GPU\n"
            f"  3. Verify JAX CUDA installation: python -c 'import jax; print(jax.devices())'\n"
            f"  4. For CPU training, use: examples/onnx/onnx-yolo-demo/\n"
        )

    print(f"✓ JAX GPU detected: {devices}")

    # Exclude shape_unsafe rewrites to prevent JAX tracer errors
    pytensor.config.optimizer_excluding = "shape_unsafe"
    print(f"✓ Optimizer exclusions set: {pytensor.config.optimizer_excluding}")

    # Use JAX mode for compilation
    pytensor.config.mode = "JAX"
    print(f"✓ PyTensor mode set to: {pytensor.config.mode}")
    print("✓ GPU-only training mode enabled")

except ImportError as e:
    raise RuntimeError(
        f"ERROR: Could not import JAX: {e}\n"
        f"Install with: uv pip install 'jax[cuda12]>=0.4.35'\n"
    )
except Exception as e:
    raise RuntimeError(
        f"ERROR: Could not configure JAX GPU backend: {e}\n"
        f"Ensure you're running on a system with NVIDIA GPU and CUDA 12 drivers.\n"
    )
```

### Success Criteria:

#### Automated Verification:
- [ ] train.sh exits on no GPU: `grep -A2 "No GPU detected" onnx-yolo-demo-gpu/scripts/train.sh | grep "exit 1"`
- [ ] train.py raises on wrong platform: `grep -A5 "device_type != \"gpu\"" onnx-yolo-demo-gpu/train.py | grep "raise RuntimeError"`
- [ ] JAX_PLATFORMS set to cuda only: `grep 'JAX_PLATFORMS="cuda"' onnx-yolo-demo-gpu/scripts/train.sh`

#### Manual Verification:
- [ ] train.sh provides helpful error on CPU-only system
- [ ] train.py immediately detects wrong JAX platform
- [ ] Error messages guide user to CPU version if needed

---

## Phase 4: Documentation and Environment Updates

### Overview
Update documentation files to emphasize Lambda Stack 22.04 requirements and create GPU-specific default environment configuration.

### Changes Required:

#### 1. Create README.md
**File**: `onnx-yolo-demo-gpu/README.md`
**Changes**: Create new file

```markdown
# YOLO11n GPU Training Demo - Lambda Stack 22.04

**GPU-ONLY Version**: This demo is configured for Lambda Stack Ubuntu 22.04 with NVIDIA GPUs.

For CPU or cross-platform training, use: `examples/onnx/onnx-yolo-demo/`

## Requirements

- **OS**: Ubuntu 22.04 (Lambda Stack recommended)
- **GPU**: NVIDIA GPU with >= 6GB VRAM
- **Driver**: NVIDIA driver >= 525.x (for CUDA 12 support)
- **CUDA**: 12.x (provided by Lambda Stack)
- **Python**: 3.11+

## Quick Start on Lambda Stack

```bash
# 1. Clone repository
git clone https://github.com/pymc-devs/pytensor.git
cd pytensor/onnx-yolo-demo-gpu

# 2. Run setup (installs dependencies, configures GPU)
bash scripts/setup.sh

# 3. Login to Weights & Biases
wandb login

# 4. Start training
bash scripts/train.sh
```

## What's Different from the CPU Version?

This GPU version differs from `examples/onnx/onnx-yolo-demo/` in these ways:

1. **Dependencies**: Uses `jax[cuda12]` instead of `jax[cpu]`
2. **GPU Required**: Scripts fail immediately without NVIDIA GPU
3. **No CPU Fallback**: JAX configured to only use CUDA platform
4. **Lambda Stack Optimized**: Setup checks for compatible drivers

## Verification

Check your GPU setup:

```bash
# Verify NVIDIA GPU
nvidia-smi

# Verify JAX sees GPU
python -c "import jax; print(jax.devices())"
# Should output: [CudaDevice(id=0)]

# Verify driver version (must be >= 525.x)
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

## Training Configuration

Default settings (adjust in scripts/train.sh):
- **Epochs**: 100
- **Batch Size**: 8 (auto-adjusted based on GPU memory)
- **Learning Rate**: 0.01
- **Image Size**: 320x320
- **Classes**: 2 (person, cellphone)

## Troubleshooting

### "No NVIDIA GPU detected"
- Verify `nvidia-smi` works
- Check you're on a GPU instance
- Use CPU version: `examples/onnx/onnx-yolo-demo/`

### "Driver version too old"
- Lambda Stack provides compatible drivers
- Manual update: `sudo apt update && sudo apt install nvidia-driver-525`

### "JAX platform is 'cpu' but must be 'gpu'"
- Reinstall JAX with CUDA: `uv pip install --upgrade 'jax[cuda12]'`
- Check `JAX_PLATFORMS=cuda` is set
- Verify CUDA libraries: `ldconfig -p | grep cuda`

## Architecture

Same model architecture as CPU version:
- **Backbone**: YOLO11n with C2PSA attention
- **Head**: FPN-PAN detection head
- **Parameters**: ~2.5M trainable parameters
- **Input**: 320x320 RGB images
- **Output**: Multi-scale predictions (P3/P4/P5)

## References

- Main PyTensor repo: `../../`
- CPU version: `examples/onnx/onnx-yolo-demo/`
- Lambda Stack docs: https://lambda.ai/lambda-stack-deep-learning-software
```

#### 2. Update .env.example
**File**: `onnx-yolo-demo-gpu/.env.example`
**Changes**: Update header and JAX section (lines 1-40)

```bash
# Environment Variables for YOLO11n GPU Training (Lambda Stack 22.04)
# Copy this file to .env and configure as needed:
#   cp .env.example .env
#
# NOTE: This GPU version requires NVIDIA GPU with CUDA 12 support

# =============================================================================
# PyTensor Configuration
# =============================================================================

# PyTensor flags (float32 required for JAX compatibility)
# optimizer_excluding=shape_unsafe prevents graph rewrites incompatible with JAX JIT
PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"

# Optional: Enable PyTensor compilation cache
# PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe,base_compiledir=~/.pytensor"

# =============================================================================
# JAX Configuration (CUDA REQUIRED)
# =============================================================================

# FORCE JAX to use GPU only (will fail if no GPU available - this is intentional)
JAX_PLATFORMS="cuda"

# Disable 64-bit floats in JAX (use float32 for better GPU performance)
JAX_ENABLE_X64=False

# Control JAX memory allocation
# Set to 'true' to preallocate GPU memory (recommended for dedicated training)
XLA_PYTHON_CLIENT_PREALLOCATE=true

# Limit GPU memory usage (0.0 to 1.0)
# Use 0.9 to leave some memory for system/display
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Enable JAX GPU memory growth (allocate as needed)
# Uncomment if you prefer dynamic allocation:
# XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Rest of file unchanged...
```

#### 3. Update QUICKSTART.md (if it exists)
**File**: `onnx-yolo-demo-gpu/QUICKSTART.md`
**Changes**: Add GPU requirement notice at the top

```markdown
# YOLO11n GPU Training Quickstart

**⚠️ GPU REQUIRED**: This version requires Lambda Stack 22.04 or Ubuntu 22.04 with NVIDIA GPU.

For CPU training, use: `examples/onnx/onnx-yolo-demo/`

---

[Rest of existing content with paths updated for root-level location]
```

### Success Criteria:

#### Automated Verification:
- [ ] README exists: `test -f onnx-yolo-demo-gpu/README.md`
- [ ] README mentions Lambda Stack: `grep "Lambda Stack" onnx-yolo-demo-gpu/README.md`
- [ ] .env.example enforces CUDA: `grep 'JAX_PLATFORMS="cuda"' onnx-yolo-demo-gpu/.env.example`
- [ ] Documentation references CPU version: `grep "examples/onnx/onnx-yolo-demo" onnx-yolo-demo-gpu/README.md`

#### Manual Verification:
- [ ] README clearly states GPU requirement
- [ ] README provides troubleshooting steps
- [ ] .env.example has appropriate defaults for GPU training
- [ ] Documentation references are accurate for root-level location

---

## Phase 5: Verification and Testing

### Overview
Verify the GPU version is correctly configured and all paths/references are updated.

### Changes Required:

#### 1. Update .gitignore (if needed)
**File**: `onnx-yolo-demo-gpu/.gitignore`
**Changes**: Ensure appropriate entries (copy from original)

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
.venv/
venv/
ENV/

# Testing
.pytest_cache/
.hypothesis/
.benchmarks/
.coverage
htmlcov/

# Training artifacts
checkpoints/
logs/
*.log
data/coco/
*.onnx
*.npz

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
```

#### 2. Create Verification Script
**File**: `onnx-yolo-demo-gpu/scripts/verify_setup.sh`
**Changes**: Create new script

```bash
#!/bin/bash
# Verify GPU setup for YOLO11n training
# Run this after setup.sh to ensure everything is configured correctly

set -e

echo "=========================================="
echo "GPU Training Environment Verification"
echo "=========================================="
echo ""

# Check 1: NVIDIA GPU
echo "[1/6] Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ FAIL: nvidia-smi not found"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "✓ PASS: NVIDIA GPU detected"
echo ""

# Check 2: Driver version
echo "[2/6] Checking driver version..."
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
if [ "$DRIVER_MAJOR" -lt 525 ]; then
    echo "❌ FAIL: Driver $DRIVER_VERSION is too old (need >= 525)"
    exit 1
fi
echo "✓ PASS: Driver $DRIVER_VERSION is compatible"
echo ""

# Check 3: Python and uv
echo "[3/6] Checking Python environment..."
if ! command -v uv &> /dev/null; then
    echo "❌ FAIL: uv not found"
    exit 1
fi
echo "✓ PASS: uv detected"
echo ""

# Check 4: JAX CUDA installation
echo "[4/6] Checking JAX installation..."
if ! uv run python -c "import jax" 2>/dev/null; then
    echo "❌ FAIL: Cannot import JAX"
    exit 1
fi
echo "✓ PASS: JAX installed"
echo ""

# Check 5: JAX GPU device
echo "[5/6] Checking JAX GPU device..."
JAX_DEVICES=$(uv run python -c "import jax; print(jax.devices())" 2>/dev/null)
if [[ "$JAX_DEVICES" != *"CudaDevice"* ]]; then
    echo "❌ FAIL: JAX not configured for GPU"
    echo "   Devices: $JAX_DEVICES"
    exit 1
fi
echo "✓ PASS: JAX configured for GPU"
echo "   $JAX_DEVICES"
echo ""

# Check 6: YOLO module imports
echo "[6/6] Checking YOLO module..."
if ! uv run python -c "from yolo.model import build_yolo11n; print('✓ Model imports successful')" 2>/dev/null; then
    echo "❌ FAIL: Cannot import YOLO model"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "Your environment is ready for GPU training."
echo "Run: bash scripts/train.sh"
echo ""
```

#### 3. Run Manual Verification
**Action**: Test file structure and imports

```bash
# Verify directory structure
ls -la onnx-yolo-demo-gpu/

# Verify pyproject.toml is valid
cd onnx-yolo-demo-gpu && uv pip check

# Verify all Python files are valid syntax
find onnx-yolo-demo-gpu -name "*.py" -exec python -m py_compile {} \;

# Check for hardcoded paths that need updating
grep -r "examples/onnx/onnx-yolo-demo" onnx-yolo-demo-gpu/ --exclude-dir=.venv
```

### Success Criteria:

#### Automated Verification:
- [ ] Verification script exists: `test -x onnx-yolo-demo-gpu/scripts/verify_setup.sh`
- [ ] No broken imports: `cd onnx-yolo-demo-gpu && uv run python -c "import yolo.model; import yolo.loss; import yolo.dataset; import yolo.blocks"`
- [ ] pyproject.toml valid: `cd onnx-yolo-demo-gpu && uv pip check`
- [ ] All scripts executable: `test -x onnx-yolo-demo-gpu/scripts/setup.sh && test -x onnx-yolo-demo-gpu/scripts/train.sh`

#### Manual Verification:
- [ ] Directory structure is complete and organized
- [ ] All references to old paths updated
- [ ] Documentation is clear and accurate
- [ ] Scripts provide helpful error messages

---

## Testing Strategy

### Unit Tests
All existing tests from `examples/onnx/onnx-yolo-demo/tests/` should work unchanged:
- `test_blocks.py` - Neural network building blocks
- `test_model_*.py` - Model architecture tests
- `test_loss.py` - Loss function tests
- `test_jax_backend.py` - JAX backend compatibility (will require GPU)

### Integration Tests
- Run test suite on Lambda Stack 22.04 instance: `uv run pytest tests/ -v`
- Verify GPU tests pass: `uv run pytest tests/ -v -m gpu`
- Test setup script on fresh Lambda Stack instance
- Test training script runs for 1-2 epochs

### Manual Testing Steps

1. **Fresh Lambda Stack Setup**:
   ```bash
   # On Lambda Stack 22.04 instance
   git clone https://github.com/pymc-devs/pytensor.git
   cd pytensor/onnx-yolo-demo-gpu
   bash scripts/setup.sh
   bash scripts/verify_setup.sh
   ```

2. **GPU Detection**:
   ```bash
   nvidia-smi  # Should show GPU
   python -c "import jax; print(jax.devices())"  # Should show CudaDevice
   ```

3. **Training Start**:
   ```bash
   bash scripts/train.sh  # Should start training immediately
   # Stop after 1-2 epochs to verify functionality
   ```

4. **Error Handling**:
   - Test setup.sh on CPU-only system (should fail gracefully)
   - Test train.sh with JAX_PLATFORMS="cpu" (should fail with clear error)
   - Test with old NVIDIA driver (should warn/fail)

## Performance Considerations

- **GPU Memory**: Default batch size of 8 requires ~4-5GB VRAM
- **CUDA Compilation**: First run will compile JAX kernels (1-2 minutes)
- **Training Speed**: Expected ~10-15 seconds per epoch on modern GPU (vs 5-10 minutes on CPU)
- **Disk Space**: COCO dataset requires ~20GB

## Migration Notes

**For users with existing CPU setup**:

1. This is a separate installation, does not affect `examples/onnx/onnx-yolo-demo/`
2. Can run both versions on same system (different directories)
3. Model checkpoints are compatible between versions
4. ONNX exports are identical (same model architecture)

**Path Changes**:
- Setup script: `cd ../` instead of `cd ../../../` to reach pytensor root
- Documentation: References updated from `examples/onnx/onnx-yolo-demo/` to `onnx-yolo-demo-gpu/`
- No changes to Python import paths (still `from yolo import ...`)

## References

- Original demo: `examples/onnx/onnx-yolo-demo/`
- Lambda Stack documentation: https://lambda.ai/lambda-stack-deep-learning-software
- JAX GPU installation: https://docs.jax.dev/en/latest/installation.html
- CUDA 12 requirements: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- PyTensor JAX backend: https://pytensor.readthedocs.io/
