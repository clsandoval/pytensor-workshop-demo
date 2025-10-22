---
date: 2025-10-18T21:43:17Z
researcher: Claude Code
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor
topic: "Adding tests for YOLO demo with Hypothesis property-based testing"
tags: [research, codebase, testing, hypothesis, yolo, onnx, demo]
status: complete
last_updated: 2025-10-18
last_updated_by: Claude Code
---

# Research: Adding Tests for YOLO Demo with Hypothesis Property-Based Testing

**Date**: 2025-10-18T21:43:17Z
**Researcher**: Claude Code
**Git Commit**: 226f34c37775b44b18b783723a7357f56fb5e116
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question

How to add tests for the YOLO demo that:
1. Follow existing testing patterns in the pytensor repository
2. Use property-based testing with Hypothesis
3. Share the same venv as the rest of the repo
4. Be placed under a `demo-tests` directory at the same level as `tests/`

## Summary

The pytensor repository has a well-structured testing framework with comprehensive patterns for property-based testing using Hypothesis. The best approach is to create a test directory within the YOLO demo following the pattern used by the existing `onnx-pymc-demo` example, which already uses Hypothesis extensively for property-based testing of ONNX operations.

**Key Findings:**
- PyTensor uses **editable installation** (`pip install -e .`) allowing subdirectories to share the main repo's venv
- **Hypothesis is actively used** in `tests/link/onnx/` with comprehensive strategies and configuration patterns
- The `onnx-pymc-demo` example already has a `tests/` directory with Hypothesis usage - serves as perfect template
- Test dependencies: pytest, hypothesis>=6.100.0, pytest-cov, pytest-benchmark, pytest-mock
- Multiple Hypothesis profiles (dev/ci/thorough) controlled via `HYPOTHESIS_PROFILE` environment variable

## Detailed Findings

### 1. YOLO Demo Location and Structure

**Location**: `examples/onnx/onnx-yolo-demo/`

**Current Structure**:
```
examples/onnx/onnx-yolo-demo/
├── .venv/                    # Local venv (can be replaced with shared)
├── yolo/
│   ├── __init__.py
│   ├── blocks.py            # Neural network building blocks
│   ├── dataset.py           # Data loading and preprocessing
│   ├── loss.py              # Loss function implementations
│   └── model.py             # YOLO model definition
├── train.py                 # Training script
├── QUICKSTART.md
└── .env.example
```

**No tests directory currently exists** - this is what needs to be added.

### 2. Existing Test Patterns in PyTensor

**Reference**: Comprehensive analysis from `tests/` directory

#### Core Testing Utilities

**unittest_tools.py** (`tests/unittest_tools.py`):
- `fetch_seed(pseed=None)`: Centralized seed management for reproducibility
- `verify_grad()`: Wrapper around gradient verification with sensible defaults
- `assert_allclose()`: Custom assertion with better error messages
- `OptimizationTestMixin`: Mixin class for testing graph optimizations
- `InferShapeTester`: Base class for testing shape inference

#### Common Test Structure

**Pattern 1: Simple Function Tests**
```python
def test_function_name():
    """Descriptive docstring."""
    # Setup
    x = vector()
    # Action
    result = some_operation(x)
    # Assertion
    assert np.allclose(result, expected)
```

**Pattern 2: Class-Based Tests**
```python
class TestFeatureName:
    def test_scenario_1(self):
        """Test specific scenario."""
        # Arrange, Act, Assert
        pass

    def test_scenario_2(self):
        """Test another scenario."""
        pass
```

**Pattern 3: Parametrized Tests**
```python
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
def test_operation(axis, size):
    """Test with multiple parameter combinations."""
    pass
```

#### Backend Comparison Pattern

All backend tests follow this pattern:
```python
def compare_backend_and_py(
    graph_inputs, graph_outputs, test_inputs, *,
    assert_fn=None, **kwargs
):
    """Compare backend output with PyTensor reference."""
    # Compile with backend
    backend_fn = function(graph_inputs, graph_outputs, mode=backend_mode)
    backend_res = backend_fn(*test_inputs)

    # Compile with Python reference
    py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = py_fn(*test_inputs)

    # Compare results
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-5)
    assert_fn(backend_res, py_res)

    return backend_fn, backend_res
```

### 3. Hypothesis Usage Patterns

**Reference**: Comprehensive analysis from `tests/link/onnx/`

#### Configuration: Multiple Profiles

**conftest.py** (`tests/link/onnx/conftest.py:10-46`):
```python
from hypothesis import HealthCheck, Phase, settings
from datetime import timedelta
import os

settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # Skip shrinking
    print_blob=False,
)

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    derandomize=True,  # Deterministic for CI
    print_blob=True,   # Print failing examples for debugging
)

settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Load profile from environment
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

**Usage:**
```bash
# Fast development (10 examples)
HYPOTHESIS_PROFILE=dev pytest tests/

# CI pipeline (100 examples, deterministic)
HYPOTHESIS_PROFILE=ci pytest tests/

# Thorough testing (1000 examples)
HYPOTHESIS_PROFILE=thorough pytest tests/
```

#### Core Strategies for Tensor Generation

**strategies/core.py** pattern (adapt for YOLO):
```python
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

def valid_shapes(min_rank=1, max_rank=4, min_dim=0, max_dim=10):
    """Generate valid tensor shapes."""
    return st.lists(
        st.integers(min_value=min_dim, max_value=max_dim),
        min_size=min_rank,
        max_size=max_rank,
    ).map(tuple)

def _safe_float_elements(dtype):
    """Generate safe float elements avoiding numerical issues."""
    if dtype in (np.float32, "float32"):
        return st.one_of(
            st.floats(min_value=-1e3, max_value=-0.1,
                     allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.1, max_value=1e3,
                     allow_nan=False, allow_infinity=False),
            st.just(0.0),
        )
    return st.floats(min_value=-1e6, max_value=1e6,
                    allow_nan=False, allow_infinity=False)

@st.composite
def yolo_tensor(draw, dtype=None, shape=None, elements=None):
    """Generate YOLO-compatible tensor."""
    if dtype is None:
        dtype = draw(st.sampled_from([np.float32, np.float64]))
    if shape is None:
        shape = draw(valid_shapes())
    if elements is None:
        elements = _safe_float_elements(dtype)

    return draw(arrays(dtype=dtype, shape=shape, elements=elements))
```

#### Property-Based Test Examples

**Pattern 1: Core Correctness Property**
```python
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture],
          deadline=None)
@given(
    op_name=st.sampled_from(list(OPERATIONS.keys())),
    data=st.data(),
)
def test_operation_correctness(tmp_path, op_name, data):
    """Property: Operation output matches reference implementation."""
    op_config = OPERATIONS[op_name]
    inputs = data.draw(op_config.input_strategy)

    # Create symbolic computation
    symbolic_inputs = [pt.tensor(f"input_{i}", dtype=inp.dtype, shape=inp.shape)
                      for i, inp in enumerate(inputs)]
    result = op_config.op_func(*symbolic_inputs)

    # Compare with reference
    compare_backend_and_py(symbolic_inputs, result, list(inputs))
```

**Pattern 2: Shape Preservation Property**
```python
@given(data=st.data())
def test_operation_shape_preservation(data):
    """Property: Operation preserves expected shape transformations."""
    input_shape = data.draw(valid_shapes(min_rank=2, max_rank=4))
    x = data.draw(yolo_tensor(shape=input_shape))

    # Apply operation
    result = some_yolo_operation(x)
    expected_shape = compute_expected_shape(input_shape)

    assert result.shape == expected_shape
```

**Pattern 3: Invariant Property**
```python
@given(
    x=yolo_tensor(shape=(3, 224, 224)),  # Fixed input size
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_yolo_detection_threshold_invariant(x, threshold):
    """Property: Higher thresholds produce fewer or equal detections."""
    detections_low = yolo_detect(x, threshold=threshold)
    detections_high = yolo_detect(x, threshold=threshold + 0.1)

    assert len(detections_high) <= len(detections_low)
```

#### Operation Registry Pattern

**Recommended for YOLO tests:**
```python
from dataclasses import dataclass
from collections.abc import Callable
from hypothesis import strategies as st

@dataclass
class YoloOperationConfig:
    """Configuration for testing a YOLO operation."""
    op_func: Callable
    input_strategy: st.SearchStrategy
    valid_dtypes: list[str]
    category: str
    notes: str | None = None

# YOLO Operation Registry
YOLO_OPERATIONS = {
    "conv_block": YoloOperationConfig(
        op_func=lambda x: yolo.blocks.ConvBlock(filters=32)(x),
        input_strategy=yolo_tensor(shape=(1, 3, 224, 224), dtype=np.float32),
        valid_dtypes=["float32"],
        category="block",
    ),
    "loss_bbox": YoloOperationConfig(
        op_func=yolo.loss.bbox_loss,
        input_strategy=bbox_prediction_ground_truth_pairs(),
        valid_dtypes=["float32"],
        category="loss",
    ),
    # ... more operations
}
```

### 4. Virtual Environment and Dependency Management

**Reference**: Analysis of `pyproject.toml` and `uv.lock`

#### Main Repo Configuration

**pyproject.toml** (Lines 48-91):
```toml
[project]
name = "pytensor"
requires-python = ">=3.11,<3.14"
dependencies = [
    "setuptools>=59.0.0",
    "scipy>=1,<2",
    "numpy>=2.0",
    # ... core dependencies
    "hypothesis>=6.140.4",
    "jax>=0.7.2", "jaxlib>=0.7.2",
]

[project.optional-dependencies]
tests = [
    "pytest",
    "pytest-cov>=2.6.1",
    "pytest-benchmark",
    "pytest-mock",
    "hypothesis>=6.100.0",
]
onnx = [
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
]
development = ["pytensor[complete]", "pytensor[tests]"]
```

**Pytest Configuration** (Lines 128-131):
```toml
[tool.pytest.ini_options]
addopts = "--durations=50 --doctest-modules"
testpaths = ["pytensor/", "tests/"]
xfail_strict = true
```

#### How Subdirectories Share the Main Venv

**Mechanism**: Editable installation

1. Install main repo in editable mode from root:
   ```bash
   cd C:\Users\armor\OneDrive\Desktop\cs\pytensor
   pip install -e ".[development,onnx]"
   # OR using uv (recommended)
   uv pip install -e ".[development,onnx]"
   ```

2. This adds `pytensor` to Python's `sys.path` from the source location

3. Subdirectories/examples can import directly:
   ```python
   import pytensor
   import pytensor.tensor as pt
   from pytensor.link.onnx import export_onnx
   ```

4. Examples can have additional dependencies via their own `pyproject.toml`

#### Root conftest.py

**conftest.py** (Lines 1-34):
- Sets `PYTENSOR_FLAGS` environment variables for all tests
- Adds `--runslow` option to control slow test execution
- Registers `@pytest.mark.slow` marker
- Automatically skips slow tests unless `--runslow` is passed

### 5. Test Directory Structure

**Reference**: Analysis of `tests/` directory

#### Main Test Directory Pattern

```
tests/                             # 207 test files total
├── unittest_tools.py              # Shared test utilities
├── test_*.py                      # Top-level tests
├── compile/
│   └── function/
├── graph/
│   ├── utils.py                   # Graph-specific test helpers
│   └── rewriting/
├── link/
│   ├── jax/
│   │   ├── conftest.py           # JAX-specific fixtures
│   │   └── test_*.py
│   ├── onnx/
│   │   ├── conftest.py           # Hypothesis config + fixtures
│   │   ├── strategies/           # Hypothesis strategies
│   │   │   ├── core.py
│   │   │   └── operations.py
│   │   └── test_*.py
│   └── pytorch/
│       ├── conftest.py
│       └── test_*.py
└── tensor/
    ├── utils.py                   # Tensor test utilities
    ├── conv/
    ├── linalg/
    └── random/
```

**Key Patterns:**
- Tests mirror source code structure
- Maximum nesting depth: 3-4 levels
- Each subdirectory can have `conftest.py` for fixtures
- `utils.py` files contain component-specific test helpers

#### Example Demo with Tests

**onnx-pymc-demo** (only example with tests currently):
```
examples/onnx/onnx-pymc-demo/
├── tests/
│   ├── conftest.py               # Demo-specific fixtures
│   ├── test_data_generation.py
│   ├── test_integration.py
│   ├── test_onnx_export.py
│   ├── test_pymc_model.py
│   └── test_pytensor_graph.py
├── pyproject.toml                # pytest config
└── [demo source files]
```

**conftest.py** (`examples/onnx/onnx-pymc-demo/tests/conftest.py`):
```python
import pytest
from hypothesis import strategies as st
import numpy as np

@pytest.fixture
def test_seed():
    """Provide consistent seed for tests."""
    return 42

@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def simple_linear_data(test_seed):
    """Generate simple linear regression data."""
    rng = np.random.default_rng(test_seed)
    X = rng.standard_normal((100, 1))
    y = 2.5 * X.squeeze() + 1.3 + rng.standard_normal(100) * 0.5
    return X, y

@st.composite
def linear_regression_data(draw, n_samples=100, n_features=1):
    """Hypothesis strategy for linear regression data."""
    # Generate random coefficients
    true_slope = draw(st.floats(min_value=-10, max_value=10))
    true_intercept = draw(st.floats(min_value=-5, max_value=5))
    noise_level = draw(st.floats(min_value=0.01, max_value=2.0))

    # Generate data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    y = true_slope * X.squeeze() + true_intercept
    y += rng.standard_normal(n_samples) * noise_level

    return X, y, (true_slope, true_intercept, noise_level)
```

**pyproject.toml** (`examples/onnx/onnx-pymc-demo/pyproject.toml`):
```toml
[project]
name = "onnx-pymc-demo"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pytensor>=2.18",    # Uses parent editable install
    "onnx>=1.14",
    "onnxruntime>=1.16",
    "pytest>=7.0",
    "hypothesis>=6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

## Recommended Implementation Plan

### Option A: Tests Within YOLO Demo (RECOMMENDED)

Follow the `onnx-pymc-demo` pattern - create tests alongside the demo:

```
examples/onnx/onnx-yolo-demo/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures and Hypothesis config
│   ├── strategies/              # Hypothesis strategies
│   │   ├── __init__.py
│   │   ├── core.py             # Core tensor/image strategies
│   │   └── yolo_data.py        # YOLO-specific strategies
│   ├── test_blocks.py          # Test yolo/blocks.py
│   ├── test_dataset.py         # Test yolo/dataset.py
│   ├── test_loss.py            # Test yolo/loss.py
│   ├── test_model.py           # Test yolo/model.py
│   └── test_integration.py     # End-to-end tests
├── yolo/
│   ├── __init__.py
│   ├── blocks.py
│   ├── dataset.py
│   ├── loss.py
│   └── model.py
├── train.py
├── pyproject.toml              # Add pytest config
└── README.md
```

**Advantages:**
- Tests live with the code they test
- Easy to find and maintain
- Self-contained demo package
- Follows existing onnx-pymc-demo pattern

### Option B: Separate demo-tests Directory

Create a separate `demo-tests/` at repo root level:

```
pytensor/
├── tests/                       # Main repo tests
├── demo-tests/                  # Demo tests (NEW)
│   ├── __init__.py
│   ├── conftest.py
│   ├── onnx/
│   │   └── yolo/
│   │       ├── conftest.py
│   │       ├── strategies/
│   │       ├── test_blocks.py
│   │       ├── test_dataset.py
│   │       ├── test_loss.py
│   │       ├── test_model.py
│   │       └── test_integration.py
│   └── pyproject.toml
├── examples/
│   └── onnx/
│       └── onnx-yolo-demo/
└── pyproject.toml
```

**Advantages:**
- Centralized location for all demo tests
- Separation between main repo tests and demo tests
- Easier to run all demo tests at once

**Disadvantages:**
- Breaks convention (onnx-pymc-demo has tests within)
- Less discoverable
- More complex import paths

### Recommended: Option A (Tests Within Demo)

## Implementation Steps

### Step 1: Create Test Directory Structure

```bash
cd examples/onnx/onnx-yolo-demo
mkdir -p tests/strategies
touch tests/__init__.py
touch tests/conftest.py
touch tests/strategies/__init__.py
touch tests/strategies/core.py
touch tests/strategies/yolo_data.py
touch tests/test_blocks.py
touch tests/test_dataset.py
touch tests/test_loss.py
touch tests/test_model.py
touch tests/test_integration.py
```

### Step 2: Configure pyproject.toml

Add to `examples/onnx/onnx-yolo-demo/pyproject.toml`:

```toml
[project]
name = "onnx-yolo-demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pytensor>=2.18",
    "numpy>=2.0",
    "jax>=0.7.2",
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "pytest>=7.0",
    "pytest-cov>=2.6.1",
    "pytest-benchmark",
    "hypothesis>=6.100.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
]
addopts = "--durations=20"
```

### Step 3: Create conftest.py with Hypothesis Config

**tests/conftest.py:**

```python
"""Pytest configuration for YOLO demo tests."""
import os
from datetime import timedelta
import pytest
import numpy as np
from hypothesis import HealthCheck, Phase, settings

# Register Hypothesis profiles
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    print_blob=False,
)

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    derandomize=True,
    print_blob=True,
)

settings.register_profile(
    "thorough",
    max_examples=500,
    deadline=None,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def test_seed():
    """Provide consistent seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(test_seed):
    """Provide NumPy random generator with consistent seed."""
    return np.random.default_rng(test_seed)


@pytest.fixture
def tmp_onnx_dir(tmp_path):
    """Create temporary directory for ONNX model files."""
    onnx_dir = tmp_path / "onnx_models"
    onnx_dir.mkdir()
    return onnx_dir


@pytest.fixture
def yolo_input_size():
    """Standard YOLO input size."""
    return (416, 416)


@pytest.fixture
def simple_image_batch(rng, yolo_input_size):
    """Generate simple batch of images for testing."""
    batch_size = 2
    channels = 3
    height, width = yolo_input_size
    return rng.standard_normal((batch_size, channels, height, width)).astype(np.float32)
```

### Step 4: Create Core Hypothesis Strategies

**tests/strategies/core.py:**

```python
"""Core Hypothesis strategies for YOLO testing."""
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np


def valid_image_shapes(
    min_size=32,
    max_size=640,
    multiple_of=32,
    fixed_size=None
):
    """Generate valid image shapes for YOLO.

    YOLO typically requires dimensions to be multiples of 32.
    """
    if fixed_size is not None:
        return st.just(fixed_size)

    valid_sizes = [i for i in range(min_size, max_size + 1, multiple_of)]
    return st.tuples(
        st.sampled_from(valid_sizes),  # height
        st.sampled_from(valid_sizes),  # width
    )


def batch_sizes(min_size=1, max_size=8):
    """Generate valid batch sizes."""
    return st.integers(min_value=min_size, max_value=max_size)


def num_channels():
    """Generate valid number of channels."""
    return st.sampled_from([1, 3])  # Grayscale or RGB


@st.composite
def yolo_image_tensor(draw, batch_size=None, channels=None, size=None, dtype=np.float32):
    """Generate YOLO-compatible image tensor.

    Returns tensor of shape (batch_size, channels, height, width).
    """
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if channels is None:
        channels = draw(num_channels())
    if size is None:
        size = draw(valid_image_shapes())

    height, width = size
    shape = (batch_size, channels, height, width)

    # Generate normalized image data [-1, 1] or [0, 1]
    elements = st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    )

    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def yolo_feature_map(draw, batch_size=None, num_filters=None, size=None):
    """Generate YOLO feature map tensor."""
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if num_filters is None:
        num_filters = draw(st.sampled_from([16, 32, 64, 128, 256, 512]))
    if size is None:
        # Feature maps are typically smaller than input
        size = draw(valid_image_shapes(min_size=13, max_size=52, multiple_of=13))

    height, width = size
    shape = (batch_size, num_filters, height, width)

    elements = st.floats(min_value=-10.0, max_value=10.0,
                        allow_nan=False, allow_infinity=False)

    return draw(arrays(dtype=np.float32, shape=shape, elements=elements))
```

**tests/strategies/yolo_data.py:**

```python
"""YOLO-specific Hypothesis strategies."""
from hypothesis import strategies as st
import numpy as np


@st.composite
def bounding_box(draw, image_size=(416, 416)):
    """Generate valid bounding box (x, y, w, h) normalized to [0, 1]."""
    # Center coordinates
    x_center = draw(st.floats(min_value=0.0, max_value=1.0))
    y_center = draw(st.floats(min_value=0.0, max_value=1.0))

    # Width and height (smaller than remaining space)
    max_width = min(2 * x_center, 2 * (1 - x_center))
    max_height = min(2 * y_center, 2 * (1 - y_center))

    width = draw(st.floats(min_value=0.01, max_value=max(0.01, max_width)))
    height = draw(st.floats(min_value=0.01, max_value=max(0.01, max_height)))

    return np.array([x_center, y_center, width, height], dtype=np.float32)


@st.composite
def yolo_detection(draw, num_classes=80, image_size=(416, 416)):
    """Generate complete YOLO detection (bbox + class + confidence)."""
    bbox = draw(bounding_box(image_size=image_size))
    class_id = draw(st.integers(min_value=0, max_value=num_classes - 1))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))

    return {
        'bbox': bbox,
        'class_id': class_id,
        'confidence': confidence,
    }


@st.composite
def yolo_ground_truth(draw, num_classes=80, max_objects=10):
    """Generate ground truth annotations for YOLO training."""
    num_objects = draw(st.integers(min_value=0, max_value=max_objects))

    detections = [draw(yolo_detection(num_classes=num_classes))
                 for _ in range(num_objects)]

    return detections
```

### Step 5: Write Property-Based Tests

**tests/test_blocks.py:**

```python
"""Property-based tests for YOLO building blocks."""
import pytest
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import pytensor.tensor as pt
from pytensor import function

from yolo import blocks
from tests.strategies.core import yolo_image_tensor, yolo_feature_map


class TestConvBlock:
    """Test ConvBlock component."""

    @given(
        input_tensor=yolo_feature_map(
            batch_size=st.just(2),
            num_filters=st.sampled_from([16, 32, 64]),
            size=st.just((52, 52))
        ),
        output_filters=st.sampled_from([32, 64, 128]),
    )
    def test_conv_block_output_shape(self, input_tensor, output_filters):
        """Property: ConvBlock produces correct output shape."""
        batch_size, in_channels, height, width = input_tensor.shape

        # Create symbolic input
        x = pt.tensor4('x')

        # Apply conv block (assuming it exists in yolo/blocks.py)
        # conv_block = blocks.ConvBlock(filters=output_filters, kernel_size=3)
        # y = conv_block(x)

        # Expected output shape (assuming stride=1, same padding)
        expected_shape = (batch_size, output_filters, height, width)

        # Compile and test
        # f = function([x], y)
        # output = f(input_tensor)
        # assert output.shape == expected_shape
        pass  # Implement based on actual blocks API

    @given(
        input_tensor=yolo_feature_map(batch_size=2, size=st.just((52, 52))),
    )
    def test_conv_block_preserves_batch_size(self, input_tensor):
        """Property: ConvBlock preserves batch dimension."""
        batch_size = input_tensor.shape[0]

        # Test implementation
        # output = apply_conv_block(input_tensor)
        # assert output.shape[0] == batch_size
        pass


class TestResidualBlock:
    """Test ResidualBlock component."""

    @given(
        input_tensor=yolo_feature_map(num_filters=st.just(64)),
    )
    def test_residual_identity(self, input_tensor):
        """Property: Residual connection preserves input shape."""
        # output = apply_residual_block(input_tensor)
        # assert output.shape == input_tensor.shape
        pass
```

**tests/test_loss.py:**

```python
"""Property-based tests for YOLO loss functions."""
import pytest
import numpy as np
from hypothesis import given, assume
from hypothesis import strategies as st

from yolo import loss
from tests.strategies.yolo_data import yolo_detection, yolo_ground_truth


class TestBBoxLoss:
    """Test bounding box loss function."""

    @given(
        pred_bbox=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=4, max_size=4
        ),
        true_bbox=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=4, max_size=4
        ),
    )
    def test_bbox_loss_non_negative(self, pred_bbox, true_bbox):
        """Property: Bounding box loss is always non-negative."""
        pred = np.array(pred_bbox, dtype=np.float32)
        true = np.array(true_bbox, dtype=np.float32)

        # loss_value = loss.bbox_loss(pred, true)
        # assert loss_value >= 0
        pass

    @given(bbox=st.lists(st.floats(min_value=0.0, max_value=1.0),
                        min_size=4, max_size=4))
    def test_bbox_loss_zero_for_identical(self, bbox):
        """Property: Loss is zero when prediction equals ground truth."""
        bbox_array = np.array(bbox, dtype=np.float32)

        # loss_value = loss.bbox_loss(bbox_array, bbox_array)
        # assert np.isclose(loss_value, 0.0, atol=1e-6)
        pass

    @given(
        bbox=st.lists(st.floats(min_value=0.0, max_value=1.0),
                     min_size=4, max_size=4),
        epsilon=st.floats(min_value=1e-4, max_value=0.1),
    )
    def test_bbox_loss_small_perturbation(self, bbox, epsilon):
        """Property: Small perturbations produce small changes in loss."""
        bbox_array = np.array(bbox, dtype=np.float32)
        perturbed = bbox_array + epsilon
        perturbed = np.clip(perturbed, 0.0, 1.0)  # Keep in valid range

        # loss_original = loss.bbox_loss(bbox_array, bbox_array)
        # loss_perturbed = loss.bbox_loss(perturbed, bbox_array)

        # Loss should increase with perturbation
        # assert loss_perturbed >= loss_original
        pass


class TestYoloLoss:
    """Test complete YOLO loss function."""

    @given(
        predictions=yolo_ground_truth(max_objects=5),
        ground_truth=yolo_ground_truth(max_objects=5),
    )
    def test_yolo_loss_properties(self, predictions, ground_truth):
        """Property: YOLO loss has expected mathematical properties."""
        # 1. Non-negative
        # loss_value = loss.yolo_loss(predictions, ground_truth)
        # assert loss_value >= 0

        # 2. Minimum at ground truth
        # loss_gt = loss.yolo_loss(ground_truth, ground_truth)
        # assert loss_gt <= loss_value
        pass
```

**tests/test_model.py:**

```python
"""Property-based tests for YOLO model."""
import pytest
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
import pytensor.tensor as pt

from yolo import model
from tests.strategies.core import yolo_image_tensor


class TestYoloModel:
    """Test YOLO model forward pass."""

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        images=yolo_image_tensor(
            batch_size=st.integers(1, 4),
            channels=st.just(3),
            size=st.just((416, 416))
        ),
    )
    def test_model_output_shape(self, images):
        """Property: Model produces correctly shaped output."""
        batch_size = images.shape[0]

        # Create model
        # yolo_model = model.YoloV3(num_classes=80)

        # Forward pass
        # output = yolo_model(images)

        # Check output shape
        # Expected: (batch_size, num_detections, 85)
        # where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        # assert output.shape[0] == batch_size
        pass

    @given(
        images=yolo_image_tensor(batch_size=2, channels=3, size=(416, 416)),
    )
    def test_model_deterministic(self, images):
        """Property: Model produces deterministic output (no dropout in eval)."""
        # yolo_model = model.YoloV3(num_classes=80)
        # yolo_model.eval()

        # output1 = yolo_model(images)
        # output2 = yolo_model(images)

        # assert np.allclose(output1, output2)
        pass
```

**tests/test_integration.py:**

```python
"""Integration tests for complete YOLO pipeline."""
import pytest
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from yolo import model, loss
from tests.strategies.core import yolo_image_tensor
from tests.strategies.yolo_data import yolo_ground_truth


class TestYoloPipeline:
    """Test complete YOLO training/inference pipeline."""

    @pytest.mark.slow
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=5,  # Integration tests can be slow
    )
    @given(
        images=yolo_image_tensor(batch_size=2, channels=3, size=(416, 416)),
        ground_truth=st.lists(
            yolo_ground_truth(max_objects=5),
            min_size=2,
            max_size=2,
        ),
    )
    def test_forward_backward_pass(self, images, ground_truth):
        """Property: Model supports forward and backward pass."""
        # yolo_model = model.YoloV3(num_classes=80)

        # Forward pass
        # predictions = yolo_model(images)

        # Compute loss
        # loss_value = loss.yolo_loss(predictions, ground_truth)

        # Check loss is finite
        # assert np.isfinite(loss_value)
        pass

    @pytest.mark.slow
    def test_onnx_export(self, tmp_onnx_dir):
        """Test: Model can be exported to ONNX."""
        # from pytensor.link.onnx import export_onnx

        # yolo_model = model.YoloV3(num_classes=80)
        # onnx_path = tmp_onnx_dir / "yolo.onnx"

        # Export to ONNX
        # export_onnx(yolo_model, onnx_path, input_shape=(1, 3, 416, 416))

        # Verify file exists
        # assert onnx_path.exists()
        pass
```

### Step 6: Install and Run Tests

```bash
# From repo root
cd C:\Users\armor\OneDrive\Desktop\cs\pytensor

# Install main repo in editable mode (if not already done)
pip install -e ".[development,onnx]"
# OR with uv (recommended)
uv pip install -e ".[development,onnx]"

# Navigate to YOLO demo
cd examples/onnx/onnx-yolo-demo

# Run tests with different profiles
HYPOTHESIS_PROFILE=dev pytest tests/ -v          # Fast (10 examples)
HYPOTHESIS_PROFILE=ci pytest tests/ -v           # Medium (50 examples)
HYPOTHESIS_PROFILE=thorough pytest tests/ -v     # Slow (500 examples)

# Run specific test file
pytest tests/test_blocks.py -v

# Run with coverage
pytest tests/ --cov=yolo --cov-report=html

# Run excluding slow tests
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

## Code References

Key files for reference:

### Test Patterns
- `tests/unittest_tools.py` - Core test utilities
- `tests/link/onnx/conftest.py:10-46` - Hypothesis configuration
- `tests/link/onnx/test_properties.py` - Property-based test examples

### Hypothesis Strategies
- `tests/link/onnx/strategies/core.py` - Tensor generation strategies
- `tests/link/onnx/strategies/operations.py` - Operation registry pattern

### Example Demo Tests
- `examples/onnx/onnx-pymc-demo/tests/conftest.py:1-154` - Fixtures and strategies
- `examples/onnx/onnx-pymc-demo/pyproject.toml` - Pytest configuration

### Configuration
- `pyproject.toml:48-91` - Project dependencies
- `pyproject.toml:128-131` - Pytest configuration
- `conftest.py:1-34` - Root test configuration

### YOLO Demo Location
- `examples/onnx/onnx-yolo-demo/yolo/` - Source code to test

## Architecture Insights

### Testing Philosophy
1. **Property-Based Testing**: Use Hypothesis to test mathematical properties and invariants rather than specific examples
2. **Backend Comparison**: Always compare custom backends against PyTensor's reference Python implementation
3. **Reproducibility**: All tests use seeded RNGs for deterministic behavior
4. **Numerical Stability**: Conservative ranges in Hypothesis strategies to avoid overflow/underflow

### Venv Sharing Pattern
1. **Editable Installation**: Main repo installed with `pip install -e .`
2. **Import Resolution**: Subdirectories import from parent via Python's `sys.path`
3. **Additional Dependencies**: Demos can have extra dependencies in their own `pyproject.toml`
4. **No Nested Venvs**: Don't create separate venvs for demos - share the main one

### Hypothesis Best Practices
1. **Multiple Profiles**: dev (fast), ci (medium), thorough (comprehensive)
2. **Composite Strategies**: Use `@st.composite` for complex data generation
3. **Operation Registry**: Centralize operation configs in a dataclass-based registry
4. **Health Check Suppression**: Suppress specific checks when needed (e.g., pytest fixtures)
5. **Conservative Ranges**: Use safe numeric ranges to avoid numerical issues

## Historical Context (from thoughts/)

No prior research documents found specifically about YOLO demo testing.

## Related Research

This is the first research document on this topic.

## Open Questions

1. **YOLO Module API**: What is the exact API of the `yolo` module? (blocks, dataset, loss, model)
   - Need to examine actual implementation to write concrete tests
   - Tests above are templates that need to be adapted

2. **GPU Testing**: How should GPU-required tests be marked and handled?
   - Consider adding `@pytest.mark.gpu` marker
   - Configure CI to skip GPU tests if hardware unavailable

3. **Test Coverage Goals**: What level of coverage is desired?
   - Unit tests for each component (blocks, loss, etc.)
   - Integration tests for complete pipeline
   - Property-based tests for mathematical properties

4. **ONNX Export Testing**: Should ONNX export be tested extensively?
   - Compare ONNX Runtime output with PyTensor output
   - Follow patterns from `tests/link/onnx/test_basic.py`

5. **Demo vs Library**: Should demo tests have same rigor as library tests?
   - Demos can have simpler tests focused on "it works" verification
   - Library tests need comprehensive coverage of edge cases
