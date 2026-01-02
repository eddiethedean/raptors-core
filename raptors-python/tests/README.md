# Raptors Python Tests

This directory contains comprehensive tests for the Raptors Python bindings.

## Test Structure

### Rust Unit Tests

Rust unit tests are located in `tests/*_test.rs` files. These tests use PyO3's testing utilities to test the Python bindings from Rust.

**Test Files:**
- `array_test.rs` - Tests for Array creation, properties, operations, and manipulation
- `dtype_test.rs` - Tests for DType creation and properties
- `ufunc_test.rs` - Tests for ufunc operations
- `iterator_test.rs` - Tests for array iteration

### Python Pytest Tests

Python tests use pytest and are located in `tests/test_*.py` files. These test the public Python API.

**Test Files:**
- `test_array.py` - Comprehensive tests for Array functionality
- `test_dtype.py` - Tests for DType functionality
- `test_ufunc.py` - Tests for ufunc operations
- `test_numpy_interop.py` - Tests for NumPy interoperability (when implemented)
- `conftest.py` - Pytest configuration and fixtures

## Running Tests

### Rust Tests

Run Rust unit tests:

```bash
cd raptors-python
cargo test
```

Run specific test:

```bash
cargo test test_array_creation
```

Run with output:

```bash
cargo test -- --nocapture
```

### Python Tests

First, ensure the Python module is built and installed:

```bash
# Build the Python extension
cd raptors-python
maturin develop

# Or if using cargo directly:
cargo build --release
# Then copy the .so/.dylib to a location in PYTHONPATH
```

Then run pytest:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_array.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=raptors
```

## Test Coverage

The test suite covers:

- ✅ Array creation (zeros, ones, empty)
- ✅ Array properties (shape, size, ndim, dtype, strides)
- ✅ Array operations (arithmetic, comparison)
- ✅ Array manipulation (reshape, transpose, copy, view)
- ✅ Array indexing (getitem, setitem)
- ✅ Array iteration
- ✅ DType creation and properties
- ✅ Ufunc operations (arithmetic, math, reductions)
- ⏳ NumPy interoperability (partially implemented)

## Adding New Tests

When adding new functionality:

1. **Add Rust tests** in `tests/*_test.rs` for testing the bindings layer
2. **Add Python tests** in `tests/test_*.py` for testing the public API
3. **Update this README** with new test coverage

### Example: Adding a Test

**Rust Test:**
```rust
#[test]
fn test_new_feature() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        // Test code here
    });
}
```

**Python Test:**
```python
def test_new_feature():
    """Test new feature"""
    import raptors
    # Test code here
    assert condition
```

