# YOLO ONNX Export Tests - TDD Implementation Plan

## Overview

Implement comprehensive Test-Driven Development (TDD) tests for ONNX export of the YOLO11n model at all architectural levels: individual building blocks, composite modules (backbone, head), and the complete integrated model. This plan addresses the current issues where (1) the YOLO demo imports system pytensor instead of the local development version, and (2) ONNX export fails with errors.

## Current State Analysis

### Current Testing Landscape:
- Testing framework: **pytest** (configured in `pyproject.toml:36-40`)
- Test directory: `examples/onnx/onnx-yolo-demo/testing/`
- Existing tests:
  - `test_model.py` - Basic model forward pass and gradient tests
  - `test_jax_components.py` - Comprehensive JAX JIT compilation tests
  - `test_jax_issues.py` - JAX-specific issue debugging
  - `test_jax_gpu_required.py` - GPU availability tests
- Available test utilities: PyTensor's `compare_onnx_and_py()` from `tests/link/onnx/test_basic.py`
- Existing test patterns: Parametrized tests, fixture-based setup, shape verification

### Current Issues:

**Issue 1: Import Configuration** (`pyproject.toml:9`)
```toml
pytensor @ file:///C:/Users/armor/OneDrive/Desktop/cs/pytensor
```
The local file dependency is configured but not being used at runtime. The demo imports system pytensor instead of the local development version.

**Issue 2: ONNX Export Failures** (`export_model.py:139`)
```python
export_onnx(
    pytensor_function=inference_fn,
    output_path=str(output_path),
)
```
ONNX export currently fails with errors. Need to identify specific failure points through systematic testing.

### YOLO11n Architecture Components:

**Building Blocks** (`yolo/blocks.py`):
1. `ConvBNSiLU` (lines 64-180) - Conv2D + BatchNorm + SiLU activation
2. `Bottleneck` (lines 182-232) - Two convolutions with residual
3. `C3k2` (lines 234-311) - CSP bottleneck with 2 convolutions
4. `SPPF` (lines 313-389) - Spatial Pyramid Pooling - Fast
5. `C2PSA` (lines 391-458) - CSP with Parallel Spatial Attention

**Model Components** (`yolo/model.py`):
1. `YOLO11nBackbone` (lines 16-119) - Feature extraction backbone
2. `YOLO11nHead` (lines 121-287) - Detection head with FPN-PAN
3. `YOLO11n` (lines 289-350) - Complete model
4. `build_yolo11n()` (lines 352-382) - Model builder function

### PyTensor ONNX Export Capabilities:

From research, PyTensor's `export_onnx()` supports:
- Conv2D with padding, strides, dilation (`dispatch/conv.py`)
- BatchNorm operations (`dispatch/batchnorm.py`)
- Pooling (max pool) (`dispatch/pool.py`)
- Elemwise ops (Add, Mul, Sigmoid, etc.) (`dispatch/elemwise.py`)
- Shape operations (Reshape, DimShuffle, Concatenate) (`dispatch/shape.py`)
- Activation functions (Softmax, SiLU via Sigmoid+Mul) (`dispatch/special.py`, `dispatch/elemwise.py`)

**Potential Issues**:
- Upsampling operation (`model.py:249-286`) uses complex dimshuffle+tile+reshape - may not convert cleanly
- Filter flipping in Conv2D (all use `filter_flip=False` which is good)
- Padding modes (all use `padding="same"` or `padding="valid"` which are supported)

## Desired End State

After implementation:
1. **Import issue resolved**: Tests run using local development pytensor
2. **Comprehensive ONNX export tests**: Every component from blocks to full model has tests
3. **Export validation**: Tests verify ONNX files are valid and structurally correct
4. **Failure identification**: Tests identify specific operations that fail ONNX export
5. **Reproducible tests**: All tests use fixed seeds for deterministic results
6. **Clear documentation**: Each test documents what it validates and why

### Success Criteria:
- [ ] All tests use local pytensor (verify via import path checks)
- [ ] Each building block has ONNX export test
- [ ] Backbone and head have separate ONNX export tests
- [ ] Full model has integration ONNX export test
- [ ] All ONNX files validated with `onnx.checker.check_model()`
- [ ] Tests identify specific failure points if export fails
- [ ] All tests are reproducible with fixed random seeds

## What We're NOT Testing/Implementing

- ONNX Runtime inference validation (just export validation)
- Numerical accuracy of ONNX outputs vs PyTensor (future phase)
- Performance benchmarks of ONNX models
- Multiple input sizes (just 320x320 for now)
- Different ONNX opset versions (use default opset 18)
- Pre-trained weight loading (use random initialization)
- Training or gradient computation through ONNX
- Web inference integration

## TDD Approach

### Test Design Philosophy:
1. **Start simple**: Test smallest components first (ConvBNSiLU)
2. **Build up**: Progress to composite modules (C3k2, SPPF)
3. **Integration last**: Test full model after parts work
4. **Fail fast**: Let failures guide us to problematic operations
5. **Validate thoroughly**: Check ONNX model structure, not just file existence

### Expected Failure Patterns:
- **NotImplementedError**: Operation not supported by PyTensor ONNX backend
- **ValueError**: Invalid ONNX model structure (shape mismatches, undefined nodes)
- **ImportError**: Module import issues (system vs local pytensor)
- **AttributeError**: Missing ONNX export functionality

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive, informative tests that attempt ONNX export at every architectural level. These tests will initially fail, revealing which operations/components cannot be exported.

### Test File Structure:

```
examples/onnx/onnx-yolo-demo/testing/
├── conftest.py              # Shared fixtures (NEW)
├── test_onnx_blocks.py      # Block-level export tests (NEW)
├── test_onnx_modules.py     # Module-level export tests (NEW)
├── test_onnx_integration.py # Full model export tests (NEW)
└── test_import_verification.py  # Import path verification (NEW)
```

---

## Test Category 1: Import Path Verification

**Test File**: `testing/test_import_verification.py`
**Purpose**: Verify that tests use local development pytensor, not system pytensor

### Test Cases:

#### Test: `test_pytensor_import_path`
**Purpose**: Verify pytensor is imported from local development directory
**Test Data**: None (import path check)
**Expected Behavior**: pytensor.__file__ points to local dev directory

```python
"""Verify that tests use local development pytensor."""
import pytensor


def test_pytensor_import_path():
    """
    Test that pytensor is imported from local development directory.

    This test ensures we're testing against the correct pytensor version
    and not accidentally using system-installed pytensor.

    Expected: pytensor.__file__ should be in:
    C:/Users/armor/OneDrive/Desktop/cs/pytensor/pytensor/
    """
    import os
    from pathlib import Path

    pytensor_file = Path(pytensor.__file__)
    expected_base = Path("C:/Users/armor/OneDrive/Desktop/cs/pytensor").resolve()
    actual_base = pytensor_file.parent.parent.resolve()

    assert actual_base == expected_base, (
        f"Wrong pytensor imported!\n"
        f"Expected base: {expected_base}\n"
        f"Actual base:   {actual_base}\n"
        f"Full path:     {pytensor_file}\n"
        f"\nThis means the test is using system pytensor instead of local dev version.\n"
        f"Fix: Run tests with 'uv run pytest' or configure PYTHONPATH."
    )

    print(f"✓ Using local pytensor from: {pytensor_file}")


def test_onnx_export_available():
    """
    Test that ONNX export functionality is available.

    Verifies that pytensor.link.onnx.export_onnx can be imported,
    which confirms ONNX dependencies are installed.
    """
    try:
        from pytensor.link.onnx import export_onnx

        # Check function signature
        import inspect
        sig = inspect.signature(export_onnx)
        assert 'pytensor_function' in sig.parameters
        assert 'output_path' in sig.parameters

        print(f"✓ ONNX export available: {export_onnx.__module__}")

    except ImportError as e:
        raise AssertionError(
            f"Cannot import export_onnx: {e}\n"
            f"This usually means:\n"
            f"1. onnx package not installed: pip install onnx\n"
            f"2. pytensor[onnx] not installed: pip install pytensor[onnx]\n"
            f"3. Using wrong pytensor version"
        ) from e
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: "Wrong pytensor imported!" with path details
- Indicates: System pytensor being used instead of local dev version

---

## Test Category 2: Block-Level ONNX Export

**Test File**: `testing/test_onnx_blocks.py`
**Purpose**: Test ONNX export of individual building blocks

### Test Cases:

#### Test: `test_convbnsilu_onnx_export`
**Purpose**: Test ConvBNSiLU block can be exported to ONNX
**Test Data**: Fixed seed (42), shape (1, 3, 32, 32)
**Expected Behavior**: Valid ONNX file created with Conv, BatchNorm, Sigmoid, Mul nodes

```python
"""Test ONNX export of YOLO building blocks."""
import os
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from pytensor import function
from pathlib import Path

# Import YOLO blocks
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo.blocks import ConvBNSiLU, Bottleneck, C3k2, SPPF, C2PSA


def test_convbnsilu_onnx_export(tmp_path):
    """
    Test that ConvBNSiLU block exports to valid ONNX.

    ConvBNSiLU = Conv2D + BatchNorm + SiLU(x) where SiLU(x) = x * sigmoid(x)

    Expected ONNX structure:
    - Conv node
    - BatchNormalization node
    - Sigmoid node
    - Mul node (for SiLU = x * sigmoid(x))

    This is the most basic building block. If this fails, more complex
    blocks will also fail.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    # Create block
    conv = ConvBNSiLU(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding="same",
        name_prefix="test_conv"
    )

    # Create symbolic computation
    x = pt.tensor4("x", dtype="float32")
    y = conv(x)

    # Compile function
    f = function([x], y)

    # Test with fixed seed data
    rng = np.random.default_rng(42)
    x_val = rng.random((1, 3, 32, 32), dtype=np.float32)

    # Verify forward pass works
    y_val = f(x_val)
    expected_shape = (1, 16, 16, 16)  # stride=2 halves spatial dims
    assert y_val.shape == expected_shape, f"Expected {expected_shape}, got {y_val.shape}"

    # Export to ONNX
    onnx_path = tmp_path / "test_convbnsilu.onnx"

    try:
        model = export_onnx(f, str(onnx_path))

        # Validate ONNX model
        onnx.checker.check_model(model)

        # Check file exists and has content
        assert onnx_path.exists(), "ONNX file not created"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"

        # Check model structure
        graph = model.graph
        node_types = [node.op_type for node in graph.node]

        print(f"\n✓ ConvBNSiLU exported successfully")
        print(f"  ONNX file: {onnx_path}")
        print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")
        print(f"  Node types: {node_types}")
        print(f"  Input shape: {x_val.shape} -> Output shape: {expected_shape}")

        # Expected nodes (order may vary)
        assert "Conv" in node_types, "Missing Conv node"
        assert "BatchNormalization" in node_types or "Mul" in node_types, (
            "Missing BatchNorm or scale/shift operations"
        )
        assert "Sigmoid" in node_types, "Missing Sigmoid node (for SiLU)"

    except Exception as e:
        print(f"\n✗ ConvBNSiLU export failed: {e}")
        print(f"  This is the simplest block. Failure here means:")
        print(f"  - Conv2D export may not work")
        print(f"  - BatchNorm export may not work")
        print(f"  - SiLU activation (x * sigmoid(x)) may not work")
        raise


def test_bottleneck_onnx_export(tmp_path):
    """
    Test that Bottleneck block exports to valid ONNX.

    Bottleneck = two ConvBNSiLU layers with optional residual connection.

    Expected: If ConvBNSiLU works, this should work too.
    Adds: Residual addition (Add node) if shortcut=True
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    # Create block (with residual)
    bottleneck = Bottleneck(
        in_channels=32,
        out_channels=32,
        shortcut=True,
        name_prefix="test_btlnk"
    )

    x = pt.tensor4("x", dtype="float32")
    y = bottleneck(x)

    f = function([x], y)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 32, 40, 40), dtype=np.float32)
    y_val = f(x_val)

    # Export
    onnx_path = tmp_path / "test_bottleneck.onnx"

    try:
        model = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model)

        assert onnx_path.exists()

        graph = model.graph
        node_types = [node.op_type for node in graph.node]

        print(f"\n✓ Bottleneck exported successfully")
        print(f"  Node types: {node_types}")
        print(f"  Has residual Add: {'Add' in node_types}")

        # Should have Add node for residual connection
        assert "Add" in node_types, "Missing Add node for residual connection"

    except Exception as e:
        print(f"\n✗ Bottleneck export failed: {e}")
        raise


def test_c3k2_onnx_export(tmp_path):
    """
    Test that C3k2 block exports to valid ONNX.

    C3k2 = CSP bottleneck with split-concatenate-merge pattern.

    Expected ONNX operations:
    - Initial 1x1 Conv (split)
    - Bottleneck blocks
    - Concatenate operation
    - Final 1x1 Conv (merge)
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    c3k2 = C3k2(
        in_channels=64,
        out_channels=64,
        n_blocks=2,
        shortcut=True,
        name_prefix="test_c3k2"
    )

    x = pt.tensor4("x", dtype="float32")
    y = c3k2(x)

    f = function([x], y)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 64, 40, 40), dtype=np.float32)

    onnx_path = tmp_path / "test_c3k2.onnx"

    try:
        model = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model)

        graph = model.graph
        node_types = [node.op_type for node in graph.node]

        print(f"\n✓ C3k2 exported successfully")
        print(f"  Node count: {len(graph.node)}")
        print(f"  Has Concat: {'Concat' in node_types}")

        assert "Concat" in node_types, "Missing Concat node for CSP pattern"

    except Exception as e:
        print(f"\n✗ C3k2 export failed: {e}")
        print(f"  C3k2 uses concatenation. This may indicate:")
        print(f"  - Concatenate operation not supported")
        print(f"  - Complex multi-path networks not supported")
        raise


def test_sppf_onnx_export(tmp_path):
    """
    Test that SPPF block exports to valid ONNX.

    SPPF = Spatial Pyramid Pooling - Fast
    Uses cascaded max pooling with concatenation.

    Expected ONNX operations:
    - MaxPool nodes (3 cascaded pools)
    - Concat node (concatenate x, y1, y2, y3)
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    sppf = SPPF(
        in_channels=256,
        out_channels=256,
        pool_size=5,
        name_prefix="test_sppf"
    )

    x = pt.tensor4("x", dtype="float32")
    y = sppf(x)

    f = function([x], y)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 256, 10, 10), dtype=np.float32)

    onnx_path = tmp_path / "test_sppf.onnx"

    try:
        model = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model)

        graph = model.graph
        node_types = [node.op_type for node in graph.node]
        maxpool_count = node_types.count("MaxPool")

        print(f"\n✓ SPPF exported successfully")
        print(f"  MaxPool count: {maxpool_count}")
        print(f"  Has Concat: {'Concat' in node_types}")

        assert maxpool_count >= 3, f"Expected 3+ MaxPool nodes, got {maxpool_count}"
        assert "Concat" in node_types, "Missing Concat node"

    except Exception as e:
        print(f"\n✗ SPPF export failed: {e}")
        print(f"  SPPF uses cascaded pooling with padding. This may indicate:")
        print(f"  - MaxPool with padding not supported")
        print(f"  - Specific padding configuration not supported")
        raise


def test_c2psa_onnx_export(tmp_path):
    """
    Test that C2PSA block exports to valid ONNX.

    C2PSA = CSP with Parallel Spatial Attention
    Similar structure to C3k2 with attention mechanism.

    Expected: Should work if C3k2 works.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    c2psa = C2PSA(
        in_channels=256,
        out_channels=256,
        name_prefix="test_c2psa"
    )

    x = pt.tensor4("x", dtype="float32")
    y = c2psa(x)

    f = function([x], y)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 256, 10, 10), dtype=np.float32)

    onnx_path = tmp_path / "test_c2psa.onnx"

    try:
        model = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model)

        print(f"\n✓ C2PSA exported successfully")

    except Exception as e:
        print(f"\n✗ C2PSA export failed: {e}")
        raise
```

**Expected Failure Modes**:
- ConvBNSiLU: May fail if Conv2D or BatchNorm conversion has issues
- Bottleneck: Should work if ConvBNSiLU works
- C3k2: May fail at Concatenate operation
- SPPF: May fail at MaxPool with padding
- C2PSA: May fail similar to C3k2

---

## Test Category 3: Module-Level ONNX Export

**Test File**: `testing/test_onnx_modules.py`
**Purpose**: Test ONNX export of composite modules (backbone, head)

### Test Cases:

#### Test: `test_backbone_onnx_export`
**Purpose**: Test YOLO11nBackbone can be exported to ONNX
**Test Data**: Fixed seed (42), shape (1, 3, 320, 320)
**Expected Behavior**: Valid ONNX file with 3 outputs (P3, P4, P5)

```python
"""Test ONNX export of YOLO composite modules."""
import os
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from pytensor import function
from pathlib import Path

# Import YOLO model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo.model import YOLO11nBackbone, YOLO11nHead


def test_backbone_onnx_export(tmp_path):
    """
    Test that YOLO11nBackbone exports to valid ONNX.

    Backbone extracts features at 3 scales from input image.
    Input: (1, 3, 320, 320)
    Outputs:
    - P3: (1, 64, 40, 40)
    - P4: (1, 128, 20, 20)
    - P5: (1, 256, 10, 10)

    This is a large graph with many operations. Success here means
    all the building blocks compose correctly for ONNX export.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    print("\n[Testing Backbone ONNX Export]")
    print("  Building backbone...")

    backbone = YOLO11nBackbone(in_channels=3)

    x = pt.tensor4("x", dtype="float32")
    p3, p4, p5 = backbone(x)

    print(f"  Compiling function...")
    f = function([x], [p3, p4, p5])

    # Test with fixed seed
    rng = np.random.default_rng(42)
    x_val = rng.random((1, 3, 320, 320), dtype=np.float32)

    print(f"  Running forward pass...")
    p3_val, p4_val, p5_val = f(x_val)

    print(f"  Input shape: {x_val.shape}")
    print(f"  P3 shape: {p3_val.shape}")
    print(f"  P4 shape: {p4_val.shape}")
    print(f"  P5 shape: {p5_val.shape}")

    # Export to ONNX
    onnx_path = tmp_path / "yolo11n_backbone.onnx"

    print(f"  Exporting to ONNX: {onnx_path}")

    try:
        model = export_onnx(f, str(onnx_path))

        # Validate
        onnx.checker.check_model(model)

        assert onnx_path.exists()
        file_size_kb = onnx_path.stat().st_size / 1024

        # Check structure
        graph = model.graph
        node_count = len(graph.node)
        output_count = len(graph.output)

        print(f"\n✓ Backbone exported successfully!")
        print(f"  File: {onnx_path}")
        print(f"  Size: {file_size_kb:.2f} KB")
        print(f"  Nodes: {node_count}")
        print(f"  Outputs: {output_count}")

        # Should have 3 outputs
        assert output_count == 3, f"Expected 3 outputs (P3, P4, P5), got {output_count}"

        # Check output shapes in ONNX
        output_shapes = []
        for output in graph.output:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            output_shapes.append(shape)

        print(f"  Output shapes: {output_shapes}")

    except Exception as e:
        print(f"\n✗ Backbone export failed: {e}")
        print(f"\nDebugging information:")
        print(f"  If this fails but all blocks passed, the issue is likely:")
        print(f"  - Composition of blocks creates unsupported pattern")
        print(f"  - Graph is too large/complex")
        print(f"  - Specific operation combination not supported")

        import traceback
        traceback.print_exc()
        raise


def test_head_onnx_export(tmp_path):
    """
    Test that YOLO11nHead exports to valid ONNX.

    Head takes backbone features and produces detection predictions.
    Uses FPN (top-down) and PAN (bottom-up) pathways.

    Key operations:
    - Upsampling (may be problematic - uses dimshuffle+tile+reshape)
    - Concatenation
    - Convolutions

    This test will reveal if upsampling is the bottleneck.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    print("\n[Testing Head ONNX Export]")
    print("  Building head...")

    head = YOLO11nHead(num_classes=2)

    # Create inputs matching backbone outputs
    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    print(f"  Compiling function...")
    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    # Test data
    rng = np.random.default_rng(42)
    p3_val = rng.random((1, 64, 40, 40), dtype=np.float32)
    p4_val = rng.random((1, 128, 20, 20), dtype=np.float32)
    p5_val = rng.random((1, 256, 10, 10), dtype=np.float32)

    print(f"  Running forward pass...")
    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    print(f"  P3: {p3_val.shape} -> {det_p3_val.shape}")
    print(f"  P4: {p4_val.shape} -> {det_p4_val.shape}")
    print(f"  P5: {p5_val.shape} -> {det_p5_val.shape}")

    # Export
    onnx_path = tmp_path / "yolo11n_head.onnx"

    print(f"  Exporting to ONNX: {onnx_path}")

    try:
        model = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model)

        assert onnx_path.exists()

        graph = model.graph
        node_types = [node.op_type for node in graph.node]

        print(f"\n✓ Head exported successfully!")
        print(f"  Nodes: {len(graph.node)}")
        print(f"  Has Upsample/Resize: {'Resize' in node_types or 'Upsample' in node_types}")

    except Exception as e:
        print(f"\n✗ Head export failed: {e}")
        print(f"\nThis is likely due to:")
        print(f"  - Upsampling operation (model.py:249-286)")
        print(f"  - Uses dimshuffle + tile + reshape pattern")
        print(f"  - May not convert cleanly to ONNX Resize/Upsample")

        import traceback
        traceback.print_exc()
        raise
```

**Expected Failure Modes**:
- Backbone: May fail if graph is too complex or specific operation combinations don't work
- Head: **Most likely to fail** due to complex upsampling operation (dimshuffle+tile+reshape)

---

## Test Category 4: Full Model Integration

**Test File**: `testing/test_onnx_integration.py`
**Purpose**: Test ONNX export of complete YOLO11n model

### Test Cases:

#### Test: `test_full_model_onnx_export`
**Purpose**: Test complete YOLO11n model end-to-end ONNX export
**Test Data**: Fixed seed (42), shape (1, 3, 320, 320)
**Expected Behavior**: Valid ONNX file matching expected structure

```python
"""Test ONNX export of complete YOLO11n model."""
import os
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from pytensor import function
from pathlib import Path

# Import YOLO
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo.model import build_yolo11n


def test_full_model_onnx_export(tmp_path):
    """
    Test that complete YOLO11n model exports to valid ONNX.

    This is the full integration test - backbone + head together.
    Input: (1, 3, 320, 320)
    Outputs:
    - det_p3: (1, 6, 40, 40)  # 6 = 4 bbox coords + 2 classes
    - det_p4: (1, 6, 20, 20)
    - det_p5: (1, 6, 10, 10)

    Success here means the entire model can be exported for deployment.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    print("\n[Testing Full YOLO11n Model ONNX Export]")
    print("=" * 70)

    # Build model
    print("\n[1/4] Building model...")
    model, x, predictions = build_yolo11n(
        num_classes=2,
        input_size=320
    )

    pred_p3, pred_p4, pred_p5 = predictions

    print(f"  Model built: {len(model.params)} parameter groups")

    # Compile function
    print("\n[2/4] Compiling PyTensor function...")
    f = function([x], [pred_p3, pred_p4, pred_p5])
    print("  Function compiled")

    # Test forward pass
    print("\n[3/4] Testing forward pass...")
    rng = np.random.default_rng(42)
    x_val = rng.random((1, 3, 320, 320), dtype=np.float32)

    p3_val, p4_val, p5_val = f(x_val)

    print(f"  Input:  {x_val.shape}")
    print(f"  Output P3: {p3_val.shape}")
    print(f"  Output P4: {p4_val.shape}")
    print(f"  Output P5: {p5_val.shape}")

    # Verify shapes
    assert p3_val.shape == (1, 6, 40, 40), f"Wrong P3 shape: {p3_val.shape}"
    assert p4_val.shape == (1, 6, 20, 20), f"Wrong P4 shape: {p4_val.shape}"
    assert p5_val.shape == (1, 6, 10, 10), f"Wrong P5 shape: {p5_val.shape}"
    print("  ✓ Shapes correct")

    # Export to ONNX
    print("\n[4/4] Exporting to ONNX...")
    onnx_path = tmp_path / "yolo11n_full.onnx"

    try:
        model_proto = export_onnx(
            pytensor_function=f,
            output_path=str(onnx_path),
            model_name="yolo11n_320"
        )

        print("  Export completed, validating...")

        # Validate ONNX model
        onnx.checker.check_model(model_proto)
        print("  ✓ ONNX validation passed")

        # Check file
        assert onnx_path.exists(), "ONNX file not created"
        file_size_kb = onnx_path.stat().st_size / 1024
        print(f"  ✓ File created: {file_size_kb:.2f} KB")

        # Analyze structure
        graph = model_proto.graph
        node_count = len(graph.node)
        input_count = len(graph.input)
        output_count = len(graph.output)
        initializer_count = len(graph.initializer)

        print(f"\n" + "=" * 70)
        print("SUCCESS! Full YOLO11n model exported to ONNX".center(70))
        print("=" * 70)
        print(f"\nModel Statistics:")
        print(f"  File: {onnx_path}")
        print(f"  Size: {file_size_kb:.2f} KB")
        print(f"  Nodes: {node_count}")
        print(f"  Inputs: {input_count}")
        print(f"  Outputs: {output_count}")
        print(f"  Initializers: {initializer_count}")

        # Node type distribution
        from collections import Counter
        node_types = [node.op_type for node in graph.node]
        type_counts = Counter(node_types)

        print(f"\nNode Type Distribution:")
        for op_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {op_type}: {count}")

        # Verify 3 outputs
        assert output_count == 3, f"Expected 3 outputs, got {output_count}"

        return model_proto

    except Exception as e:
        print(f"\n" + "=" * 70)
        print("FAILED: Full model ONNX export failed".center(70))
        print("=" * 70)
        print(f"\nError: {e}")
        print(f"\nDiagnostics:")
        print(f"  Forward pass: ✓ (works)")
        print(f"  Export call: ✗ (failed)")
        print(f"\nLikely causes:")
        print(f"  1. Upsampling operation in head (model.py:249-286)")
        print(f"  2. Complex dimshuffle+tile+reshape pattern")
        print(f"  3. Some operation combination not supported")
        print(f"\nTo debug further:")
        print(f"  - Check if backbone-only export works")
        print(f"  - Check if head-only export works")
        print(f"  - If head fails, upsampling is the issue")

        import traceback
        traceback.print_exc()
        raise


def test_full_model_with_batch_size(tmp_path):
    """
    Test ONNX export with batch size > 1.

    Verifies that the model handles batching correctly in ONNX.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    print("\n[Testing ONNX Export with Batch Size 2]")

    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    pred_p3, pred_p4, pred_p5 = predictions

    f = function([x], [pred_p3, pred_p4, pred_p5])

    # Batch size = 2
    rng = np.random.default_rng(42)
    x_val = rng.random((2, 3, 320, 320), dtype=np.float32)

    p3_val, p4_val, p5_val = f(x_val)

    print(f"  Input: {x_val.shape}")
    print(f"  P3: {p3_val.shape}, P4: {p4_val.shape}, P5: {p5_val.shape}")

    # Export
    onnx_path = tmp_path / "yolo11n_batch2.onnx"

    try:
        model_proto = export_onnx(f, str(onnx_path))
        onnx.checker.check_model(model_proto)

        print(f"  ✓ Batch size 2 export successful")

    except Exception as e:
        print(f"  ✗ Batch size 2 export failed: {e}")
        raise


def test_different_num_classes(tmp_path):
    """
    Test ONNX export with different number of classes.

    Verifies that the model structure adapts correctly to different
    numbers of output classes.
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    print("\n[Testing ONNX Export with Different Class Counts]")

    for num_classes in [1, 2, 10, 80]:  # COCO has 80 classes
        print(f"\n  Testing with {num_classes} classes...")

        model, x, predictions = build_yolo11n(
            num_classes=num_classes,
            input_size=320
        )
        pred_p3, pred_p4, pred_p5 = predictions

        f = function([x], [pred_p3, pred_p4, pred_p5])

        rng = np.random.default_rng(42)
        x_val = rng.random((1, 3, 320, 320), dtype=np.float32)
        p3_val, p4_val, p5_val = f(x_val)

        expected_channels = 4 + num_classes  # bbox + classes
        assert p3_val.shape[1] == expected_channels, (
            f"Expected {expected_channels} channels, got {p3_val.shape[1]}"
        )

        # Export
        onnx_path = tmp_path / f"yolo11n_classes{num_classes}.onnx"

        try:
            model_proto = export_onnx(f, str(onnx_path))
            onnx.checker.check_model(model_proto)
            print(f"    ✓ {num_classes} classes: success")

        except Exception as e:
            print(f"    ✗ {num_classes} classes: failed - {e}")
            if num_classes == 2:
                # If 2 classes fails, others will too
                raise
```

**Expected Failure Modes**:
- Full model: **Most comprehensive test** - reveals all issues
- Batch size test: May fail if batch dimension handling has issues
- Class count test: Should work if full model works

---

## Test Category 5: Shared Fixtures

**Test File**: `testing/conftest.py`
**Purpose**: Shared fixtures for all ONNX export tests

```python
"""Shared fixtures for ONNX export tests."""
import os
import pytest
from pathlib import Path
import sys

# Set PyTensor flags before any imports
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def setup_python_path(project_root):
    """
    Add project root to Python path for imports.

    This ensures `from yolo.model import ...` works.
    """
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield
    # Cleanup
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_exports")


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def verify_pytensor_path():
    """
    Verify that we're using local development pytensor.

    Runs once at session start to catch import issues early.
    """
    import pytensor
    from pathlib import Path

    pytensor_file = Path(pytensor.__file__)
    expected_base = Path("C:/Users/armor/OneDrive/Desktop/cs/pytensor").resolve()
    actual_base = pytensor_file.parent.parent.resolve()

    if actual_base != expected_base:
        pytest.fail(
            f"Wrong pytensor version!\n"
            f"Expected: {expected_base}\n"
            f"Actual:   {actual_base}\n"
            f"\nFix: Run tests with 'uv run pytest testing/'"
        )

    print(f"\n✓ Using local pytensor: {pytensor_file}")
```

---

## Phase 1 Success Criteria

### Automated Verification:
- [ ] All test files created with proper structure
- [ ] Tests use pytest fixtures correctly
- [ ] Tests can be collected: `pytest --collect-only testing/test_onnx_*.py`
- [ ] All imports resolve correctly (no ImportError during collection)
- [ ] Tests use fixed random seeds (reproducible)

### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Expected failure modes are documented
- [ ] Test code is readable and follows patterns from PyTensor tests

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and document how they fail. This reveals which specific operations cannot be exported to ONNX and guides implementation.

### Verification Steps:

1. **Run import verification first**:
   ```bash
   cd examples/onnx/onnx-yolo-demo
   uv run pytest testing/test_import_verification.py -v
   ```

2. **If import test fails**: Fix Python path issue before proceeding

3. **Run block-level tests**:
   ```bash
   uv run pytest testing/test_onnx_blocks.py -v --tb=short
   ```

4. **Run module-level tests**:
   ```bash
   uv run pytest testing/test_onnx_modules.py -v --tb=short
   ```

5. **Run integration tests**:
   ```bash
   uv run pytest testing/test_onnx_integration.py -v --tb=short
   ```

6. **Document all failures** in a failure matrix

### Expected Failures:

Create a failure tracking document:

| Test | Expected Result | Actual Result | Error Type | Root Cause |
|------|----------------|---------------|------------|------------|
| test_pytensor_import_path | FAIL (wrong path) | | AssertionError | uv not using local pytensor |
| test_convbnsilu_onnx_export | PASS or FAIL | | | Conv2D/BatchNorm issue |
| test_bottleneck_onnx_export | PASS if ConvBNSiLU passes | | | |
| test_c3k2_onnx_export | PASS or FAIL at Concat | | | |
| test_sppf_onnx_export | PASS or FAIL at MaxPool | | | |
| test_c2psa_onnx_export | PASS if C3k2 passes | | | |
| test_backbone_onnx_export | FAIL if any blocks fail | | | Composition issue |
| test_head_onnx_export | **LIKELY FAIL** | | NotImplementedError | Upsampling operation |
| test_full_model_onnx_export | FAIL if backbone or head fails | | | Depends on above |

### Success Criteria:

#### Automated Verification:
- [ ] All tests run and are discovered: `pytest --collect-only`
- [ ] Test execution completes (no hangs): `pytest --tb=line`
- [ ] Failure messages are captured: `pytest -v > test_failures.txt 2>&1`

#### Manual Verification:
- [ ] Each test failure is documented with error type
- [ ] Each test failure is documented with root cause hypothesis
- [ ] Failure patterns are identified (e.g., "all tests fail at upsampling")
- [ ] No unexpected test errors (syntax errors, import errors after setup)
- [ ] Failure messages point to specific operations
- [ ] Stack traces are informative

### Adjustment Phase:

Based on failures:
- [ ] If import test fails: Fix PYTHONPATH or uv configuration
- [ ] If all tests fail immediately: Fix environment setup
- [ ] If specific operation fails consistently: Document as known limitation
- [ ] If upsampling fails: This is the expected bottleneck to fix

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Fix the issues revealed by failing tests, one component at a time. Work from simplest to most complex.

### Implementation Strategy:

**Order of Implementation:**
1. Fix import/environment issues (test_import_verification)
2. Fix block-level exports (test_onnx_blocks.py)
3. Fix module-level exports (test_onnx_modules.py)
4. Fix full model export (test_onnx_integration.py)

### Implementation 1: Fix Import Path Issue

**Target Test**: `test_pytensor_import_path`
**Current Failure**: Using system pytensor instead of local dev version

**Changes Required:**

**File**: Run tests with proper Python path configuration

**Option A - Use uv run** (Recommended):
```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest testing/ -v
```

**Option B - Set PYTHONPATH**:
```bash
export PYTHONPATH="C:/Users/armor/OneDrive/Desktop/cs/pytensor:$PYTHONPATH"
pytest testing/ -v
```

**Option C - Update pyproject.toml**:
```toml
[tool.pytest.ini_options]
pythonpath = ["../../.."]
```

**Debugging Approach:**
1. Run: `uv run python -c "import pytensor; print(pytensor.__file__)"`
2. Verify output is: `C:\Users\armor\OneDrive\Desktop\cs\pytensor\pytensor\__init__.py`
3. If wrong: Check uv.lock has correct pytensor entry
4. If still wrong: Run `uv sync` to rebuild environment

**Success Criteria:**

##### Automated Verification:
- [ ] Test passes: `uv run pytest testing/test_import_verification.py::test_pytensor_import_path -v`
- [ ] Import path is correct
- [ ] ONNX export function is available

##### Manual Verification:
- [ ] Can import yolo modules without error
- [ ] PyTensor version is from local directory
- [ ] All ONNX dependencies are available

### Implementation 2: Fix Block-Level Export Issues

**Target Tests**: All tests in `test_onnx_blocks.py`
**Current Failure**: TBD (depends on Phase 2 results)

**Potential Issues & Solutions:**

**Issue 2A: Conv2D Export Fails**

If `test_convbnsilu_onnx_export` fails at Conv2D:

**Root Cause**: Conv2D parameters not supported
**Solution**: Check padding mode and stride configuration
**File**: `yolo/blocks.py:163-169`

```python
# Current code:
if self.padding == "same":
    pad_h = (self.kernel_size - 1) // 2
    pad_w = (self.kernel_size - 1) // 2
    border_mode = (pad_h, pad_w)
```

**Verification**: Run isolated Conv2D test with same parameters

**Issue 2B: BatchNorm Export Fails**

If failure is at BatchNorm:

**Root Cause**: JAX-compatible BatchNorm may not convert to ONNX
**Solution**: May need to modify `jax_compatible_batch_norm` in `blocks.py:22-62`
**Investigation**: Check if standard PyTensor BatchNorm works instead

**Issue 2C: SiLU Activation Fails**

If failure is at SiLU (x * sigmoid(x)):

**Root Cause**: Composite activation may need explicit SiLU op
**Solution**: Check if PyTensor has SiLU op, or if Sigmoid+Mul pattern converts
**File**: `blocks.py:176-177`

**Issue 2D: Concatenate Fails (C3k2, SPPF, C2PSA)**

If failure is at concatenation:

**Root Cause**: Concatenate along channel axis may have issues
**Solution**: Verify concatenation parameters are correct
**File**: `blocks.py:307`, `blocks.py:385`, `blocks.py:454`

**Success Criteria:**

##### Automated Verification:
- [ ] ConvBNSiLU test passes: `pytest testing/test_onnx_blocks.py::test_convbnsilu_onnx_export -v`
- [ ] Bottleneck test passes
- [ ] C3k2 test passes
- [ ] SPPF test passes
- [ ] C2PSA test passes
- [ ] All blocks export to valid ONNX

##### Manual Verification:
- [ ] Each ONNX file can be opened with `onnx.load()`
- [ ] Node types match expected operations
- [ ] No structural errors in exported graphs

### Implementation 3: Fix Upsampling in Head

**Target Test**: `test_head_onnx_export`
**Current Failure**: **Most likely failure point** - upsampling operation

**Root Cause**: Complex upsampling implementation in `model.py:249-286`

Current implementation:
```python
def _upsample(self, x, scale=2):
    """Upsample using nearest neighbor (JAX-compatible)."""
    # (B, C, H, W) -> (B, C, H, 1, W, 1)
    x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")

    # (B, C, H, 1, W, 1) -> (B, C, H, scale, W, scale)
    x_tiled = pt.tile(x_expanded, (1, 1, 1, scale, 1, scale))

    # (B, C, H, scale, W, scale) -> (B, C, H, W, scale, scale)
    x_rearranged = x_tiled.dimshuffle(0, 1, 2, 4, 3, 5)

    # (B, C, H, W, scale, scale) -> (B, C, H*scale, W*scale)
    x_upsampled = x_rearranged.reshape(
        (batch_size, channels, out_height, out_width)
    )

    return x_upsampled
```

**Problem**: This complex pattern may not convert to ONNX Resize/Upsample node

**Solution Options:**

**Option A: Use PyTensor resize operation** (if available):
```python
from pytensor.tensor.extra_ops import repeat

def _upsample(self, x, scale=2):
    """Upsample using repeat operation."""
    # Repeat along height
    x_h = repeat(x, scale, axis=2)
    # Repeat along width
    x_hw = repeat(x_h, scale, axis=3)
    return x_hw
```

**Option B: Use explicit Resize op** (if PyTensor has ONNX-compatible resize):
```python
# Check if pytensor.tensor.nnet.abstract_conv has resize/upsample
```

**Option C: Rewrite using simpler operations**:
```python
def _upsample(self, x, scale=2):
    """
    Simplified upsampling that may convert better to ONNX.
    Uses repeat which maps more directly to ONNX Upsample.
    """
    from pytensor.tensor.extra_ops import repeat

    # Get shape
    batch_size = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    # Repeat elements
    # (B, C, H, W) -> (B, C, H*scale, W)
    x = repeat(x, scale, axis=2)
    # (B, C, H*scale, W) -> (B, C, H*scale, W*scale)
    x = repeat(x, scale, axis=3)

    return x
```

**File**: `yolo/model.py:249-286`

**Debugging Approach:**
1. Test isolated upsampling operation with ONNX export
2. Try different upsampling implementations
3. Check PyTensor ONNX dispatch for Resize/Upsample support
4. If no support, may need to add ONNX converter for this pattern

**Success Criteria:**

##### Automated Verification:
- [ ] Head test passes: `pytest testing/test_onnx_modules.py::test_head_onnx_export -v`
- [ ] Exported ONNX has Resize or Upsample nodes
- [ ] ONNX validation succeeds

##### Manual Verification:
- [ ] Upsampling operation converts cleanly
- [ ] ONNX graph structure is reasonable
- [ ] Output shapes are correct in ONNX

### Implementation 4: Fix Full Model Export

**Target Test**: `test_full_model_onnx_export`
**Current Failure**: Depends on backbone and head results

**Changes Required:**

If backbone and head both export successfully, full model should work.

**Potential Integration Issues:**

**Issue 4A: Graph too large**
- Symptom: Export hangs or runs out of memory
- Solution: May need to optimize graph or export in pieces

**Issue 4B: Shared variables not handled**
- Symptom: Error about shared variables or initializers
- Solution: Ensure all model parameters are properly registered

**Issue 4C: Multiple outputs handling**
- Symptom: Error about output format
- Solution: Verify tuple output format is correct

**Debugging Approach:**
1. Ensure backbone and head tests pass first
2. Run full model test
3. If fails, check if issue is in composition or individual components
4. Verify parameter count matches expectations

**Success Criteria:**

##### Automated Verification:
- [ ] Full model test passes: `pytest testing/test_onnx_integration.py::test_full_model_onnx_export -v`
- [ ] ONNX file created and validated
- [ ] File size is reasonable (should be several MB)
- [ ] All batch size and class count tests pass

##### Manual Verification:
- [ ] Can load ONNX file in external tools (Netron, onnxruntime)
- [ ] Graph structure looks correct
- [ ] All parameters are included as initializers
- [ ] Input/output shapes are correct

### Complete Feature Implementation:

Once all individual tests pass:

**Final Integration:**
- Run full test suite: `uv run pytest testing/test_onnx_*.py -v`
- Verify no regressions in existing tests: `uv run pytest testing/ -v`
- Test export_model.py script: `uv run python export_model.py`

**Success Criteria:**

##### Automated Verification:
- [ ] All ONNX export tests pass: `pytest testing/test_onnx_*.py -v`
- [ ] No regressions in existing tests: `pytest testing/ -v`
- [ ] export_model.py script works: `uv run python export_model.py`
- [ ] ONNX file created at expected location

##### Manual Verification:
- [ ] Can load exported ONNX in Netron for visualization
- [ ] ONNX model structure matches expected architecture
- [ ] Export script provides informative output
- [ ] No errors or warnings during export

---

## Phase 4: Refactoring & Cleanup

### Overview
Now that tests are green and export works, refactor test code and implementation for maintainability.

### Refactoring Targets:

1. **Test Code Duplication**:
   - Extract common export validation logic
   - Create helper for "export and validate ONNX"
   - Consolidate seed and data generation

2. **Test Organization**:
   - Ensure consistent test naming
   - Improve docstrings with examples
   - Add parametrized tests for variations

3. **Implementation Code**:
   - Document any workarounds for ONNX export
   - Add comments explaining ONNX-specific choices
   - Ensure upsampling implementation is clean

4. **Export Script**:
   - Add more informative output
   - Better error messages
   - Validation after export

### Refactoring Steps:

#### Refactoring 1: Extract Common Test Utilities

**File**: `testing/test_utils.py` (NEW)

```python
"""Common utilities for ONNX export tests."""
import numpy as np
from pathlib import Path
from typing import Callable, List, Tuple


def export_and_validate_onnx(
    pytensor_function: Callable,
    output_path: Path,
    test_input: np.ndarray,
    expected_output_shapes: List[Tuple[int, ...]],
) -> "onnx.ModelProto":
    """
    Export PyTensor function to ONNX and validate.

    Parameters
    ----------
    pytensor_function : callable
        Compiled PyTensor function
    output_path : Path
        Where to save ONNX file
    test_input : np.ndarray
        Test input for forward pass
    expected_output_shapes : list of tuple
        Expected shapes of outputs

    Returns
    -------
    onnx.ModelProto
        Validated ONNX model

    Raises
    ------
    AssertionError
        If export fails or validation fails
    """
    from pytensor.link.onnx import export_onnx
    import onnx

    # Run forward pass to verify function works
    if isinstance(test_input, np.ndarray):
        outputs = pytensor_function(test_input)
    else:
        outputs = pytensor_function(*test_input)

    # Verify shapes
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    assert len(outputs) == len(expected_output_shapes), (
        f"Output count mismatch: {len(outputs)} vs {len(expected_output_shapes)}"
    )

    for i, (output, expected_shape) in enumerate(zip(outputs, expected_output_shapes)):
        assert output.shape == expected_shape, (
            f"Output {i} shape mismatch: {output.shape} vs {expected_shape}"
        )

    # Export to ONNX
    model = export_onnx(pytensor_function, str(output_path))

    # Validate
    onnx.checker.check_model(model)

    # Verify file
    assert output_path.exists(), f"ONNX file not created at {output_path}"
    assert output_path.stat().st_size > 0, "ONNX file is empty"

    return model


def create_fixed_seed_data(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Create reproducible random data with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def print_onnx_summary(model: "onnx.ModelProto", file_path: Path):
    """Print summary of ONNX model structure."""
    from collections import Counter

    graph = model.graph
    node_types = [node.op_type for node in graph.node]
    type_counts = Counter(node_types)

    print(f"\nONNX Model Summary:")
    print(f"  File: {file_path}")
    print(f"  Size: {file_path.stat().st_size / 1024:.2f} KB")
    print(f"  Nodes: {len(graph.node)}")
    print(f"  Inputs: {len(graph.input)}")
    print(f"  Outputs: {len(graph.output)}")
    print(f"  Initializers: {len(graph.initializer)}")
    print(f"\n  Top node types:")
    for op_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {op_type}: {count}")
```

Then refactor tests to use these utilities.

#### Refactoring 2: Improve Export Script

**File**: `export_model.py`

Add better validation after export:

```python
# After export (line 144):
print(f"✓ ONNX model exported successfully!")
print(f"✓ File size: {output_path.stat().st_size / 1024:.2f} KB")
print(f"✓ Location: {output_path.absolute()}\n")

# Add validation
print("Validating exported model...")
import onnx
model_check = onnx.load(str(output_path))
onnx.checker.check_model(model_check)
print("✓ ONNX model validation passed")

# Print model info
graph = model_check.graph
print(f"\nModel Information:")
print(f"  Nodes: {len(graph.node)}")
print(f"  Inputs: {len(graph.input)}")
print(f"  Outputs: {len(graph.output)}")

# Print output shapes
print(f"\nOutput Shapes:")
for output in graph.output:
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
             for dim in output.type.tensor_type.shape.dim]
    print(f"  {output.name}: {shape}")
```

#### Refactoring 3: Add Documentation

**File**: `testing/README.md` (NEW)

```markdown
# YOLO ONNX Export Tests

This directory contains comprehensive tests for ONNX export of the YOLO11n model.

## Test Organization

- `test_import_verification.py` - Verify correct pytensor version
- `test_onnx_blocks.py` - Test individual building blocks
- `test_onnx_modules.py` - Test backbone and head modules
- `test_onnx_integration.py` - Test complete model export
- `conftest.py` - Shared fixtures
- `test_utils.py` - Common utilities

## Running Tests

Run all ONNX export tests:
```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest testing/test_onnx_*.py -v
```

Run specific test file:
```bash
uv run pytest testing/test_onnx_blocks.py -v
```

Run specific test:
```bash
uv run pytest testing/test_onnx_integration.py::test_full_model_onnx_export -v
```

## Test Coverage

Each level of the architecture has tests:
- Building blocks: ConvBNSiLU, Bottleneck, C3k2, SPPF, C2PSA
- Modules: Backbone, Head
- Integration: Full YOLO11n model

All tests verify ONNX export succeeds and produces valid ONNX files.

## Expected Output

Successful test run exports ONNX files to temporary directory and validates:
- File is created and non-empty
- ONNX structure passes validation
- Expected operations are present
- Output shapes are correct
```

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass after refactoring: `pytest testing/test_onnx_*.py -v`
- [ ] No regressions: `pytest testing/ -v`
- [ ] Export script still works: `uv run python export_model.py`

#### Manual Verification:
- [ ] Test code is more readable
- [ ] No code duplication in tests
- [ ] Documentation is clear and helpful
- [ ] Export script provides better output
- [ ] Code follows project conventions

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] All 5 building blocks: Individual ONNX export tests
- [ ] Both modules: Backbone and Head ONNX export tests
- [ ] Full integration: Complete model ONNX export test
- [ ] Edge cases: Batch sizes, class counts
- [ ] Import verification: Correct pytensor version

### Test Organization:
- Test files: `testing/test_onnx_*.py`
- Fixtures: `testing/conftest.py`
- Utilities: `testing/test_utils.py`
- Documentation: `testing/README.md`

### Running Tests:

```bash
# All ONNX export tests
uv run pytest testing/test_onnx_*.py -v

# Specific test file
uv run pytest testing/test_onnx_blocks.py -v

# Specific test
uv run pytest testing/test_onnx_blocks.py::test_convbnsilu_onnx_export -v

# With detailed output
uv run pytest testing/test_onnx_*.py -vv --tb=short

# With full tracebacks
uv run pytest testing/test_onnx_*.py -vv --tb=long
```

## Performance Considerations

ONNX export can be slow for large models:
- Block-level tests: < 5 seconds each
- Module tests: 10-30 seconds each
- Full model test: 30-60 seconds
- Total test time: ~2-5 minutes

### Performance Testing:
Not included in this phase, but could add:
- [ ] Benchmark export time for each component
- [ ] Compare ONNX Runtime inference vs PyTensor
- [ ] Memory usage during export

## Migration Notes

This adds new test files but does not modify existing tests:
- Existing tests (`test_model.py`, `test_jax_components.py`) remain unchanged
- New tests are additive, not replacing existing tests
- Export functionality is tested separately from model functionality

## References

- Original ticket: Inline in command
- PyTensor ONNX export: `C:\Users\armor\OneDrive\Desktop\cs\pytensor\pytensor\link\onnx\export.py`
- PyTensor ONNX tests: `C:\Users\armor\OneDrive\Desktop\cs\pytensor\tests\link\onnx\`
- YOLO blocks: `examples/onnx/onnx-yolo-demo/yolo/blocks.py`
- YOLO model: `examples/onnx/onnx-yolo-demo/yolo/model.py`
- Export script: `examples/onnx/onnx-yolo-demo/export_model.py`
