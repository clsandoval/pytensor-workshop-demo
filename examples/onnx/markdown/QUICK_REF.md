# ONNX Backend Quick Reference

## Setup (One-time)

```bash
# Option 1: Automated setup
bash setup_onnx_dev.sh

# Option 2: Manual setup
uv venv
source .venv/Scripts/activate
uv pip install -e ".[development,onnx]"
pre-commit install
```

## Daily Workflow

```bash
# Activate venv
source .venv/Scripts/activate

# Make changes to code
# ...

# Run tests
pytest tests/link/onnx/ -v

# Check style
pre-commit run --all-files

# Commit
git add .
git commit -m "feat(onnx): implement basic elemwise ops"
```

## Essential Commands

### Testing
```bash
pytest tests/link/onnx/ -v              # Run ONNX tests
pytest -k "onnx" -v                     # Run tests matching "onnx"
pytest tests/link/onnx/test_basic.py::test_export_simple_add -v  # Specific test
pytest --lf                             # Re-run last failures
```

### Code Quality
```bash
ruff format .                           # Format code
ruff check --fix .                      # Fix issues
pre-commit run --all-files              # Run all hooks
```

### Development
```bash
ipython                                 # Interactive Python
pytensor-cache clear                    # Clear cache
python test_onnx_quick.py               # Quick smoke test
```

## File Locations

| What | Where |
|------|-------|
| Implementation | `pytensor/link/onnx/` |
| Tests | `tests/link/onnx/` |
| Examples | `examples/onnx_demo/` |
| Research | `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md` |

## Common Patterns

### Adding a new Op conversion

```python
# In pytensor/link/onnx/dispatch/elemwise.py

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.math import YourOp

@onnx_funcify.register(YourOp)
def onnx_funcify_YourOp(op, node, var_names, get_var_name, **kwargs):
    from onnx import helper

    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return helper.make_node(
        "ONNXOpName",
        inputs=input_names,
        outputs=output_names,
        name=f"ONNXOpName_{output_names[0]}"
    )
```

### Writing a test

```python
# In tests/link/onnx/test_elemwise.py

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

def test_your_op():
    """Test YourOp conversion."""
    x = pt.vector('x', dtype='float32')
    y = pt.your_op(x)

    f = pytensor.function([x], y)

    # Export to ONNX
    model = export_onnx(f, "/tmp/test.onnx")

    # Validate with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession("/tmp/test.onnx")

    x_val = np.array([1, 2, 3], dtype='float32')
    result = session.run(None, {'x': x_val})
    expected = f(x_val)

    np.testing.assert_allclose(result[0], expected)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `uv pip install -e ".[development,onnx]"` |
| Tests failing | `pytest tests/link/onnx/ -vv -s` |
| C compile errors | `export PYTENSOR_FLAGS='cxx='` |
| Cache issues | `pytensor-cache clear` |
| Style failures | `pre-commit run --all-files` |

## Important URLs

- Implementation Plan: `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md`
- PyTensor Docs: https://pytensor.readthedocs.io/
- ONNX Operators: https://github.com/onnx/onnx/blob/main/docs/Operators.md
- ONNX Python API: https://onnx.ai/onnx/api/

## Phase 1 Checklist

- [ ] Create directory structure
- [ ] Implement `pytensor/link/onnx/__init__.py`
- [ ] Implement `pytensor/link/onnx/linker.py`
- [ ] Implement `pytensor/link/onnx/dispatch/basic.py`
- [ ] Implement `pytensor/link/onnx/export.py`
- [ ] Write basic tests
- [ ] Test with simple ops (Add, Mul)
- [ ] Validate with ONNX Runtime
