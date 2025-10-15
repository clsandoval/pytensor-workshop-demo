# ONNX Backend Development Guide

This guide will help you set up your local development environment for building the ONNX backend for PyTensor.

## Quick Start

```bash
# 1. Run the setup script
bash setup_onnx_dev.sh

# 2. Activate the virtual environment
source .venv/Scripts/activate  # Git Bash
# OR
.venv\Scripts\Activate.ps1     # PowerShell

# 3. Verify installation
python -c "import pytensor, onnx, onnxruntime; print('✓ All dependencies installed!')"
```

## Manual Setup (if you prefer)

### 1. Create Virtual Environment with UV

```bash
# Create venv
uv venv

# Activate
source .venv/Scripts/activate  # Git Bash/Linux/Mac
# OR
.venv\Scripts\Activate.ps1     # PowerShell
```

### 2. Install Dependencies

```bash
# Install PyTensor in editable mode with all dependencies
uv pip install -e ".[development,onnx]"

# Or step by step:
uv pip install -e "."                    # Core PyTensor
uv pip install -e ".[tests]"             # Testing tools
uv pip install -e ".[jax]"               # JAX backend (optional, for reference)
uv pip install -e ".[numba]"             # Numba backend (optional, for reference)
uv pip install onnx>=1.14.0              # ONNX core
uv pip install onnxruntime>=1.16.0       # ONNX Runtime for testing
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatters and linters before each commit.

### 4. Verify Installation

```bash
# Check PyTensor
python -c "import pytensor; print(pytensor.__version__)"

# Check ONNX
python -c "import onnx; print(onnx.__version__)"

# Check ONNX Runtime
python -c "import onnxruntime as ort; print(ort.__version__)"

# Quick functional test
python -c "
import pytensor
import pytensor.tensor as pt
import numpy as np

x = pt.vector('x')
y = x + 1
f = pytensor.function([x], y)
print('PyTensor result:', f([1, 2, 3]))
print('✓ PyTensor is working!')
"
```

## Directory Structure

Your ONNX backend will live here:

```
pytensor/
├── link/
│   ├── jax/           # Reference: JAX backend
│   ├── numba/         # Reference: Numba backend
│   └── onnx/          # NEW: Your ONNX backend
│       ├── __init__.py
│       ├── linker.py          # ONNXLinker class
│       ├── export.py          # export_onnx() function
│       └── dispatch/
│           ├── __init__.py
│           ├── basic.py       # Core onnx_funcify dispatcher
│           ├── elemwise.py    # Elemwise ops
│           ├── nlinalg.py     # Matrix ops
│           └── special.py     # Activations

tests/
├── link/
│   └── onnx/          # NEW: ONNX tests
│       ├── __init__.py
│       ├── test_basic.py
│       ├── test_elemwise.py
│       └── test_export.py

examples/
└── onnx_demo/         # NEW: WebAssembly demo
    ├── train_model.py
    ├── index.html
    └── README.md
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b onnx-backend-initial
```

### 2. Implement a Component

Start with the core infrastructure (see your implementation plan):

1. `pytensor/link/onnx/__init__.py` - Public API
2. `pytensor/link/onnx/linker.py` - ONNXLinker class
3. `pytensor/link/onnx/dispatch/basic.py` - Core dispatcher
4. `pytensor/link/onnx/export.py` - Export function

### 3. Write Tests

```bash
# Create test file
touch tests/link/onnx/test_basic.py

# Run tests
pytest tests/link/onnx/test_basic.py -v

# Run with coverage
pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=html
```

### 4. Check Code Style

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Or just on staged files
pre-commit run

# Format with ruff (automatically done by pre-commit)
ruff format pytensor/link/onnx/

# Check for issues
ruff check pytensor/link/onnx/
```

### 5. Test Your Changes

```bash
# Run ONNX tests only
pytest tests/link/onnx/ -v

# Run all tests (takes longer)
pytest

# Run specific test
pytest tests/link/onnx/test_basic.py::test_export_simple_add -v

# Run with more output
pytest tests/link/onnx/ -vv -s
```

## Common Development Commands

### Testing

```bash
# Run all ONNX tests
pytest tests/link/onnx/ -v

# Run with coverage
pytest tests/link/onnx/ --cov=pytensor.link.onnx --cov-report=term-missing

# Run specific test file
pytest tests/link/onnx/test_basic.py -v

# Run specific test function
pytest tests/link/onnx/test_basic.py::test_export_simple_add -v

# Run tests matching pattern
pytest -k "onnx" -v

# Show print statements
pytest tests/link/onnx/ -v -s

# Stop at first failure
pytest tests/link/onnx/ -x

# Run in parallel (faster)
pytest tests/link/onnx/ -n auto
```

### Code Quality

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Type checking (optional)
python scripts/run_mypy.py

# Run pre-commit hooks
pre-commit run --all-files
```

### Interactive Development

```bash
# Start IPython with PyTensor
ipython

# In IPython:
>>> import pytensor
>>> import pytensor.tensor as pt
>>> from pytensor.link.onnx import export_onnx
>>>
>>> # Test your code
>>> x = pt.vector('x')
>>> y = x + 1
>>> f = pytensor.function([x], y)
>>> f([1, 2, 3])
```

### Debugging

```bash
# Run with debugger
pytest tests/link/onnx/test_basic.py --pdb

# Set breakpoint in code:
import pdb; pdb.set_trace()

# Or use ipdb for better interface:
pip install ipdb
import ipdb; ipdb.set_trace()
```

### Cache Management

```bash
# Clear PyTensor cache (useful after C compilation issues)
pytensor-cache clear

# List cache contents
pytensor-cache list

# Show cache location
python -c "import pytensor; print(pytensor.config.compiledir)"
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the virtual environment
which python  # Should point to .venv/Scripts/python or .venv/bin/python

# Reinstall in editable mode
uv pip install -e ".[development,onnx]"
```

### C Compilation Issues (Windows)

```bash
# PyTensor needs a C compiler for some backends
# Install Microsoft Visual C++ Build Tools or use Python-only backends

# Force Python-only mode for testing
export PYTENSOR_FLAGS='cxx='
```

### ONNX Import Errors

```bash
# Reinstall ONNX packages
uv pip install --upgrade onnx onnxruntime

# Check installation
python -c "import onnx; import onnxruntime; print('OK')"
```

### Test Failures

```bash
# Run with more verbose output
pytest tests/link/onnx/ -vv -s

# Show full error traceback
pytest tests/link/onnx/ --tb=long

# Run only failed tests from last run
pytest --lf
```

## Useful References

### Existing Backends (for reference)

- **JAX Backend**: `pytensor/link/jax/` - Simple, clean implementation
- **Numba Backend**: `pytensor/link/numba/` - More complex with custom vectorization
- **C Backend**: `pytensor/link/c/` - Most complex, generates C code

### Study These Files

1. `pytensor/link/jax/linker.py` - How JAXLinker works
2. `pytensor/link/jax/dispatch/basic.py` - jax_funcify dispatcher
3. `pytensor/link/jax/dispatch/elemwise.py` - Elemwise conversion example
4. `pytensor/link/utils.py` - Utility functions for linkers

### ONNX Documentation

- **ONNX Python API**: https://onnx.ai/onnx/api/
- **ONNX Operators**: https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **ONNX Runtime**: https://onnxruntime.ai/docs/
- **ONNX Opset Versions**: https://github.com/onnx/onnx/blob/main/docs/Versioning.md

## Next Steps

1. **Read the Implementation Plan**: `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md`
2. **Study JAX Backend**: Look at how it's structured for reference
3. **Start with Phase 1**: Implement core infrastructure files
4. **Write Tests First**: TDD approach works well here
5. **Test Incrementally**: Test each op conversion as you implement it

## Getting Help

- **PyTensor Docs**: https://pytensor.readthedocs.io/
- **PyTensor GitHub**: https://github.com/pymc-devs/pytensor
- **ONNX Docs**: https://onnx.ai/
- **Your Research**: `thoughts/shared/research/` directory

## Quick Test Script

Save this as `test_onnx_quick.py`:

```python
"""Quick test script for ONNX backend development."""

import numpy as np
import pytensor
import pytensor.tensor as pt

# Test 1: Basic computation
print("Test 1: Basic PyTensor function")
x = pt.vector('x', dtype='float32')
y = pt.vector('y', dtype='float32')
z = x + y * 2

f = pytensor.function([x, y], z)
result = f([1, 2, 3], [4, 5, 6])
print(f"  Result: {result}")
print(f"  Expected: {[9., 12., 15.]}")
print(f"  ✓ Pass" if np.allclose(result, [9., 12., 15.]) else "  ✗ Fail")

# Test 2: ONNX export (once implemented)
print("\nTest 2: ONNX export")
try:
    from pytensor.link.onnx import export_onnx
    model = export_onnx(f, "/tmp/test_model.onnx")
    print("  ✓ ONNX export working!")
except ImportError:
    print("  ℹ ONNX export not yet implemented")
except Exception as e:
    print(f"  ⚠ Error: {e}")

print("\n✓ Quick test complete!")
```

Run with:
```bash
python test_onnx_quick.py
```

Good luck with your ONNX backend implementation! 🚀
