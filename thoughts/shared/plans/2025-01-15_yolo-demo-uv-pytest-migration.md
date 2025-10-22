# YOLO Demo: UV Migration and Pytest Test Suite Implementation Plan

## Overview

Migrate the `examples/onnx/onnx-yolo-demo` project from traditional pip/venv dependency management to `uv` and convert all existing test files to a proper pytest test suite.

## Current State Analysis

The YOLO demo currently uses:
- **Dependency Management**: `requirements.txt` + `pip install`
- **Virtual Environment**: Standard Python `venv`
- **Testing**: Manual test scripts in `testing/` directory run as standalone Python scripts
- **Script Execution**: Direct `python` commands in shell scripts

### Project Structure:
```
examples/onnx/onnx-yolo-demo/
├── requirements.txt          # Dependencies
├── scripts/
│   ├── setup.sh             # Environment setup with venv + pip
│   └── train.sh             # Training script using `python`
├── testing/
│   ├── test_jax_issues.py   # Manual test runner
│   ├── test_model.py        # Manual test runner
│   └── test_jax_components.py  # Manual test runner with argparse
├── train.py                 # Main training script
├── QUICKSTART.md            # User documentation
└── yolo/                    # Model implementation
```

### Current Dependencies (from requirements.txt):
- numpy>=1.24.0
- scipy>=1.10.0
- jax[cuda12]>=0.4.20
- pillow>=10.0.0
- tqdm>=4.65.0
- pycocotools>=2.0.6
- requests>=2.31.0
- wandb>=0.15.0
- pyyaml>=6.0
- PyTensor (editable install from parent directory)

### Key Discoveries:
- All test files are currently run as scripts with `if __name__ == "__main__"` blocks
- `test_jax_components.py` uses argparse for `--jax` and `--quick` flags
- Setup script creates venv and uses pip for all installations
- Train script activates venv and runs `python train.py`
- PyTensor is installed as editable: `pip install -e ../../../`

## Desired End State

After this migration:
1. **Dependency management**: All dependencies managed via `pyproject.toml` and `uv`
2. **No requirements.txt**: Removed entirely
3. **Script execution**: All Python scripts run via `uv run python`
4. **Testing**: All tests in `testing/` directory run via `uv run pytest`
5. **Test format**: All tests converted to pytest-compatible format
6. **Documentation**: QUICKSTART.md updated to reflect uv usage

### Success Criteria:

#### Automated Verification:
- [ ] `uv sync` completes successfully
- [ ] `uv run pytest testing/ -v` runs all tests
- [ ] All tests pass with pytest
- [ ] `uv run python train.py --help` works
- [ ] No `requirements.txt` file exists
- [ ] JAX detects GPU: `uv run python -c "import jax; assert jax.devices()[0].platform == 'gpu'"`

#### Manual Verification:
- [ ] Setup workflow documented in QUICKSTART.md works end-to-end
- [ ] Training can be initiated with `uv run python train.py`
- [ ] Tests can be run individually with `uv run pytest testing/test_model.py -v`
- [ ] PyTensor editable install still works correctly
- [ ] JAX GPU forcing works: `uv run python train.py` shows "✓ JAX GPU detected"
- [ ] Environment variables are properly set in .env file

## What We're NOT Doing

- NOT migrating to `tests/` directory (keeping `testing/`)
- NOT adding pytest markers or test categorization
- NOT changing test logic or test coverage
- NOT modifying the actual YOLO model implementation
- NOT changing PyTensor or JAX configuration
- NOT adding new tests (only converting existing ones)
- NOT creating separate test configs for different scenarios

## Implementation Approach

The migration will be done in three phases:
1. **Phase 1**: Create `pyproject.toml` and update dependency management
2. **Phase 2**: Convert test files to pytest format
3. **Phase 3**: Update scripts and documentation

## Phase 1: Create pyproject.toml and Setup UV

### Overview
Create the `pyproject.toml` file to replace `requirements.txt` and enable uv-based dependency management.

### Changes Required:

#### 1. Create pyproject.toml
**File**: `examples/onnx/onnx-yolo-demo/pyproject.toml` (NEW)
**Changes**: Create new file with project metadata and dependencies

```toml
[project]
name = "yolo11n-pytensor"
version = "0.1.0"
description = "YOLO11n implementation using PyTensor with JAX backend"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "jax[cuda12]>=0.4.20",
    "pillow>=10.0.0",
    "tqdm>=4.65.0",
    "pycocotools>=2.0.6",
    "requests>=2.31.0",
    "wandb>=0.15.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-xdist>=3.0.0",  # parallel test execution
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["testing"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

#### 2. Update scripts/setup.sh
**File**: `examples/onnx/onnx-yolo-demo/scripts/setup.sh`
**Changes**: Replace pip/venv commands with uv equivalents, ensure JAX GPU forcing

**Key modifications:**
- Remove venv creation steps (lines 68-80)
- Remove pip upgrade (lines 84-86)
- Replace `pip install -e . --quiet` with `uv pip install -e . --quiet`
- Replace `pip install --upgrade "jax[cuda12]"...` with dependency managed by pyproject.toml
- Replace `pip install --quiet numpy scipy...` with `uv sync`
- Add `uv pip install -e ../../../` for PyTensor editable install
- **CRITICAL**: Add `JAX_PLATFORMS="cuda,cpu"` to .env file to force JAX GPU usage
- Update success messages to reference uv instead of pip

**Before (lines 89-112):**
```bash
# Install PyTensor with JAX backend
echo "[7/8] Installing PyTensor and dependencies..."
echo "This may take several minutes..."

# Install from the parent pytensor directory
cd ../../../
pip install -e . --quiet

# Install JAX with CUDA support
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
```

**After:**
```bash
# Install dependencies with uv
echo "[7/8] Installing dependencies with uv..."
echo "This may take several minutes..."

# Install PyTensor from parent directory (editable)
cd ../../../
uv pip install -e .

# Return to demo directory
cd examples/onnx/onnx-yolo-demo/

# Sync all dependencies from pyproject.toml
uv sync

# Install dev dependencies
uv pip install -e ".[dev]"
```

**Also update .env file creation (lines 126-145):**

**Before:**
```bash
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
```

**After:**
```bash
cat > .env << 'ENVEOF'
# PyTensor Configuration
PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"

# JAX Platform Configuration - FORCE GPU USAGE
JAX_PLATFORMS="cuda,cpu"
JAX_ENABLE_X64=False

# JAX GPU Memory Configuration
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# WandB Configuration
WANDB_PROJECT=yolo11n-pytensor

# System Configuration
PYTHONUNBUFFERED=1
ENVEOF
```

#### 3. Delete requirements.txt
**File**: `examples/onnx/onnx-yolo-demo/requirements.txt`
**Changes**: Delete this file entirely

### Success Criteria:

#### Automated Verification:
- [ ] `pyproject.toml` exists and is valid TOML
- [ ] `bash scripts/setup.sh` completes without errors
- [ ] `uv sync` installs all dependencies correctly
- [ ] `uv pip list` shows all required packages
- [ ] PyTensor is installed in editable mode
- [ ] `.env` file contains `JAX_PLATFORMS="cuda,cpu"`

#### Manual Verification:
- [ ] JAX can detect GPU: `uv run python -c "import jax; print(jax.devices())"`
- [ ] JAX platform is GPU: `uv run python -c "import jax; print(jax.devices()[0].platform)"`
- [ ] PyTensor imports work: `uv run python -c "import pytensor; print(pytensor.__version__)"`
- [ ] All YOLO modules import: `uv run python -c "from yolo.model import build_yolo11n"`
- [ ] Environment variables are exported: `cat .env | grep JAX_PLATFORMS`

---

## Phase 2: Convert Tests to Pytest

### Overview
Convert all three test files in `testing/` directory to pytest-compatible format.

### Changes Required:

#### 1. Create pytest configuration fixture file
**File**: `examples/onnx/onnx-yolo-demo/testing/conftest.py` (NEW)
**Changes**: Create shared fixtures for JAX setup with GPU forcing

```python
"""Pytest configuration and shared fixtures for YOLO tests."""

import os
import pytest
import sys

# CRITICAL: Set environment variables BEFORE importing JAX or PyTensor
# These force JAX to prioritize GPU over CPU
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"

import pytensor


@pytest.fixture(scope="session")
def jax_setup():
    """Setup JAX backend if available, forcing GPU usage."""
    try:
        import jax

        # Force JAX to use GPU by checking devices
        devices = jax.devices()
        device_type = devices[0].platform if len(devices) > 0 else "none"

        if device_type == "gpu":
            # Only enable JAX mode if GPU is available
            pytensor.config.mode = "JAX"
            pytensor.config.optimizer_excluding = "shape_unsafe"

            return {
                "jax_available": True,
                "devices": devices,
                "device_type": device_type,
                "mode": pytensor.config.mode
            }
        else:
            # JAX available but no GPU - tests will skip
            return {
                "jax_available": False,
                "devices": devices,
                "device_type": device_type,
                "error": f"JAX found but no GPU detected. Platform: {device_type}",
                "mode": pytensor.config.mode
            }
    except Exception as e:
        return {
            "jax_available": False,
            "error": str(e),
            "mode": pytensor.config.mode
        }


@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """Ensure parent directory is in Python path for imports."""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
```

#### 2. Convert test_jax_issues.py to pytest
**File**: `examples/onnx/onnx-yolo-demo/testing/test_jax_issues.py`
**Changes**: Convert to pytest format

**Modifications:**
- Remove `if __name__ == "__main__"` block (lines 262-263)
- Remove `main()` function (lines 223-260)
- Remove duplicate JAX setup code (lines 22-31) - use fixture instead
- Add pytest imports at top
- Keep all individual test functions as-is (they already start with `test_`)
- Use `jax_setup` fixture in tests that need JAX

**Example conversion for one test:**

**Before:**
```python
def test_repeat_operation():
    """Test if pt.repeat works with JAX JIT."""
    print("\n[Test 1] pt.repeat operation")
    try:
        x = pt.tensor4("x", dtype="float32")
        # ... test logic ...
        print("  ✓ pt.repeat works!")
        return True
    except Exception as e:
        print(f"  ✗ pt.repeat FAILS: {e}")
        return False
```

**After:**
```python
def test_repeat_operation(jax_setup):
    """Test if pt.repeat works with JAX JIT."""
    if not jax_setup["jax_available"]:
        pytest.skip("JAX not available")

    x = pt.tensor4("x", dtype="float32")
    # ... test logic ...
    # pytest will automatically handle assertion failures
```

#### 3. Convert test_model.py to pytest
**File**: `examples/onnx/onnx-yolo-demo/testing/test_model.py`
**Changes**: Convert to pytest format

**Modifications:**
- Remove `if __name__ == "__main__"` block (lines 184-185)
- Remove `main()` function (lines 159-182)
- Remove `sys.path.insert(0, "..")` (line 17) - use conftest fixture
- Add pytest imports
- Keep all test functions as-is (already start with `test_`)
- Replace `assert` statements to use pytest's better error messages

**Example:**

**Before:**
```python
def test_conv_bn_silu():
    """Test ConvBNSiLU block."""
    print("\n[Test 1/5] Testing ConvBNSiLU block...")
    # ... test code ...
    expected_shape = (1, 16, 160, 160)
    assert y_val.shape == expected_shape, (
        f"Expected {expected_shape}, got {y_val.shape}"
    )
    print(f"  ✓ ConvBNSiLU output shape: {y_val.shape}")
```

**After:**
```python
def test_conv_bn_silu():
    """Test ConvBNSiLU block."""
    # ... test code ...
    expected_shape = (1, 16, 160, 160)
    assert y_val.shape == expected_shape, (
        f"Expected {expected_shape}, got {y_val.shape}"
    )
```

#### 4. Convert test_jax_components.py to pytest
**File**: `examples/onnx/onnx-yolo-demo/testing/test_jax_components.py`
**Changes**: Convert to pytest format, remove argparse

**Modifications:**
- Remove `if __name__ == "__main__"` block (lines 676-677)
- Remove `main()` function with argparse (lines 595-673)
- Remove `setup_jax()` function (lines 35-61) - use fixture
- Remove argparse imports (line 15)
- Add pytest imports
- Remove `--jax` and `--quick` argument handling
- Use `jax_setup` fixture for all tests
- Keep all 13 test functions as-is

**Key changes:**

**Before (main function):**
```python
def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test YOLO11n components with JAX JIT")
    parser.add_argument("--jax", action="store_true", help="Enable JAX mode")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()

    jax_enabled = setup_jax(args.jax)
    # ... run tests ...
```

**After (removed):**
```python
# No main function needed - pytest discovers and runs all test_* functions
```

**Before (test function):**
```python
def test_backbone():
    """Test YOLO11n backbone."""
    print("\n[Test 8] YOLO11n Backbone")
    try:
        # ... test code ...
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
```

**After (test function):**
```python
def test_backbone(setup_python_path):
    """Test YOLO11n backbone."""
    from yolo.model import YOLO11nBackbone
    # ... test code ...
    # Pytest automatically captures exceptions and reports failures
```

### Success Criteria:

#### Automated Verification:
- [ ] `uv run pytest testing/ -v` discovers all tests
- [ ] All test files can be imported without errors
- [ ] Each test function is properly discovered by pytest
- [ ] `uv run pytest testing/test_model.py -v` runs only test_model tests
- [ ] `uv run pytest testing/test_jax_issues.py -v` runs only test_jax_issues tests
- [ ] `uv run pytest testing/test_jax_components.py -v` runs only test_jax_components tests
- [ ] Test output shows proper pytest formatting

#### Manual Verification:
- [ ] Tests can be run individually: `uv run pytest testing/test_model.py::test_conv_bn_silu -v`
- [ ] Failed tests show clear pytest error messages
- [ ] JAX fixture properly skips tests when JAX not available
- [ ] All original test logic is preserved

---

## Phase 3: Update Scripts and Documentation

### Overview
Update all shell scripts and documentation to use `uv run` instead of direct `python` commands.

### Changes Required:

#### 1. Update scripts/train.sh
**File**: `examples/onnx/onnx-yolo-demo/scripts/train.sh`
**Changes**: Replace `python` commands with `uv run python`, ensure JAX GPU forcing

**Modifications:**
- Line 14-23: Keep .env file loading (JAX_PLATFORMS must be exported!)
- Line 25-33: Remove venv activation, add uv check
- Line 99: Change `python dataset.py` to `uv run python dataset.py`
- Line 112-123: Keep JAX environment variable exports (CRITICAL for GPU)
- Line 125: Change `python train.py` to `uv run python train.py`
- Update success messages

**Before (lines 25-33):**
```bash
# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found. Run setup.sh first!"
    exit 1
fi
```

**After:**
```bash
# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠ uv not found. Install with: pip install uv"
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓ uv detected"
```

**CRITICAL - Keep environment variable exports (lines 112-123):**
```bash
# CRITICAL: Set PyTensor and JAX flags BEFORE Python starts
# This must happen in the shell, not inside Python
# JAX_PLATFORMS forces JAX to prioritize CUDA over CPU
export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
export JAX_PLATFORMS="cuda,cpu"
export JAX_ENABLE_X64=False
echo "PyTensor config: $PYTENSOR_FLAGS"
echo "JAX platforms: $JAX_PLATFORMS"
echo ""
```

**Before (line 99):**
```bash
python dataset.py --data-dir ./data/coco --split train
```

**After (line 99):**
```bash
uv run python dataset.py --data-dir ./data/coco --split train
```

**Before (line 125):**
```bash
python train.py \
    --epochs $EPOCHS \
    # ... arguments ...
```

**After (line 125):**
```bash
uv run python train.py \
    --epochs $EPOCHS \
    # ... arguments ...
```

**Note**: The `uv run` command will inherit the exported environment variables, ensuring JAX uses GPU.

#### 2. Update train.py docstring
**File**: `examples/onnx/onnx-yolo-demo/train.py`
**Changes**: Update usage documentation (line 12)

**Before:**
```python
Usage:
    PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run" python train.py --epochs 100 --batch-size 8 --lr 0.01
```

**After:**
```python
Usage:
    PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run" uv run python train.py --epochs 100 --batch-size 8 --lr 0.01
```

#### 3. Update QUICKSTART.md
**File**: `examples/onnx/onnx-yolo-demo/QUICKSTART.md`
**Changes**: Replace all pip/venv/python references with uv equivalents

**Key changes throughout the file:**

**Line 13-14 (setup command):**
```markdown
# Before:
bash setup.sh

# After:
bash scripts/setup.sh
```

**Line 17-20 (remove WandB venv activation):**
```markdown
# Before:
# 4. (Optional) Login to WandB for training visualization
source venv/bin/activate
wandb login

# After:
# 4. (Optional) Login to WandB for training visualization
wandb login
```

**Line 24 (training command):**
```markdown
# Before:
nohup bash train.sh > training.log 2>&1 &

# After:
nohup bash scripts/train.sh > training.log 2>&1 &
```

**Line 91 (resume training):**
```markdown
# Before:
source venv/bin/activate
python train.py --resume checkpoints/interrupted_checkpoint.npz

# After:
uv run python train.py --resume checkpoints/interrupted_checkpoint.npz
```

**Line 124-131 (remove virtual environment references):**
```markdown
# Before:
## File Sizes

- Virtual environment: ~2GB
- COCO dataset (optional): ~20GB train, ~1GB val

# After:
## File Sizes

- UV cache: ~500MB
- COCO dataset (optional): ~20GB train, ~1GB val
```

**Add new section after line 64:**
```markdown
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
```

#### 4. Update scripts/clean_cache.sh (if it exists)
**File**: `examples/onnx/onnx-yolo-demo/scripts/clean_cache.sh`
**Changes**: Add uv cache cleaning if needed

Check if this file references venv or pip cache, and update accordingly.

### Success Criteria:

#### Automated Verification:
- [ ] `bash scripts/train.sh` starts training without errors
- [ ] `uv run python train.py --help` shows help message
- [ ] All Python commands in scripts work with `uv run`
- [ ] No references to `source venv/bin/activate` remain
- [ ] No references to bare `python` commands remain (except in comments)

#### Manual Verification:
- [ ] QUICKSTART.md instructions are accurate and complete
- [ ] Training can be started following QUICKSTART.md instructions
- [ ] Tests can be run following QUICKSTART.md instructions
- [ ] No broken links or outdated references in documentation
- [ ] Setup workflow works on a fresh clone of the repo

---

## Testing Strategy

### Unit Tests (all in testing/ directory):
- `test_model.py`: Basic model architecture tests (5 tests)
  - ConvBNSiLU block functionality
  - C3k2 block functionality
  - SPPF block functionality
  - Full YOLO11n forward pass
  - Gradient computation

- `test_jax_issues.py`: JAX JIT compatibility tests (6 tests)
  - pt.repeat operation
  - Dictionary returns
  - Ellipsis slicing
  - Simplified YOLO head with upsample
  - Full YOLO11n model
  - Training step with gradients

- `test_jax_components.py`: Comprehensive component tests (13 tests)
  - Basic operations
  - Dimshuffle and tile
  - Upsampling operation
  - ConvBNSiLU block
  - Bottleneck block
  - C3k2 block
  - SPPF block
  - Backbone
  - Detection head
  - Full model
  - Loss function
  - Gradients
  - Training step with updates

### Running Tests:
```bash
# All tests
uv run pytest testing/ -v

# Specific file
uv run pytest testing/test_model.py -v

# Specific test
uv run pytest testing/test_model.py::test_conv_bn_silu -v

# With output
uv run pytest testing/ -v -s
```

## Performance Considerations

- **UV caching**: uv caches packages, making subsequent installs faster
- **Editable install**: PyTensor remains editable for development
- **Test execution**: Pytest runs tests in process, slightly faster than manual scripts
- **Parallel testing**: pytest-xdist is included but not required

## Migration Notes

### For Existing Users:

If you have an existing setup with venv:
1. Delete the old `venv/` directory: `rm -rf venv/`
2. Install uv: `pip install uv` (or use system package manager)
3. Run new setup: `bash scripts/setup.sh`
4. Run tests: `uv run pytest testing/ -v`

### Backward Compatibility:

This change is NOT backward compatible with the old setup. Users must:
- Install uv
- Remove old venv
- Re-run setup script

### UV Installation:

Users need to install uv before running setup:
```bash
# Option 1: Via pip
pip install uv

# Option 2: Via curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 3: Via package manager
# macOS: brew install uv
# Linux: see https://github.com/astral-sh/uv#installation
```

## JAX GPU Forcing - Critical Implementation Details

### The Problem
JAX may default to CPU if not explicitly configured, even when a GPU is available. This is especially problematic when:
- Environment variables aren't set before JAX is imported
- Using `uv run` without proper environment variable propagation
- The `JAX_PLATFORMS` variable isn't configured

### The Solution (Multi-layer approach)

#### Layer 1: .env file (setup.sh)
The setup script creates a `.env` file with:
```bash
JAX_PLATFORMS="cuda,cpu"      # Forces JAX to prioritize CUDA
JAX_ENABLE_X64=False          # Disable 64-bit for performance
PYTENSOR_FLAGS="...,optimizer_excluding=shape_unsafe"  # Fixes JIT issues
```

#### Layer 2: Shell script exports (train.sh)
The training script **explicitly exports** these variables:
```bash
export JAX_PLATFORMS="cuda,cpu"
export JAX_ENABLE_X64=False
export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
```

These exports happen **before** `uv run python` is called, ensuring they're inherited.

#### Layer 3: Test configuration (conftest.py)
For pytest, we set environment variables **at the top of conftest.py**:
```python
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["JAX_ENABLE_X64"] = "False"
```

This happens **before** any imports of JAX or PyTensor.

#### Layer 4: Runtime verification (train.py)
The train.py script checks if GPU is actually detected:
```python
devices = jax.devices()
device_type = devices[0].platform if len(devices) > 0 else "none"

if device_type == "gpu":
    pytensor.config.mode = "JAX"
    # ... training with GPU
else:
    # ... warning, fallback to CPU
```

### Verification Steps

After setup, verify JAX GPU forcing:

```bash
# Check JAX can see GPU
uv run python -c "import jax; print('Devices:', jax.devices()); print('Platform:', jax.devices()[0].platform)"

# Expected output:
# Devices: [cuda(id=0)]
# Platform: gpu

# Check environment variables are set
echo $JAX_PLATFORMS  # Should show: cuda,cpu

# Run tests with GPU
uv run pytest testing/test_jax_components.py -v -s
# Should show: "✓ JAX GPU detected: [cuda(id=0)]"
```

### Common Issues and Fixes

**Issue**: JAX still uses CPU despite environment variables
**Fix**: Ensure `JAX_PLATFORMS` is exported **before** Python starts:
```bash
export JAX_PLATFORMS="cuda,cpu"
uv run python train.py
```

**Issue**: Tests skip with "JAX not available"
**Fix**: Check that conftest.py sets `os.environ["JAX_PLATFORMS"]` at the top

**Issue**: Train.py shows "JAX device platform: cpu"
**Fix**: Verify CUDA is installed and `nvidia-smi` works, then ensure environment variables are exported

## References

- UV documentation: https://github.com/astral-sh/uv
- Pytest documentation: https://docs.pytest.org/
- JAX GPU configuration: https://jax.readthedocs.io/en/latest/gpu_configuration.html
- Original setup: `examples/onnx/onnx-yolo-demo/scripts/setup.sh`
- Current tests: `examples/onnx/onnx-yolo-demo/testing/`
