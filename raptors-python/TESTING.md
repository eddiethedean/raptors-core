# Testing Guide for Raptors Python

This guide explains how to run and write tests for the Raptors Python bindings.

## Quick Start

### Run All Tests

```bash
# Using the test runner script (recommended)
./run_tests.sh

# Or using Make
make test
```

This will:
1. Check if the module is built
2. Build it if necessary
3. Run Rust unit tests
4. Run Python pytest tests

## Test Structure

### Rust Unit Tests

Located in `tests/*_test.rs`. These test the Python bindings from Rust using PyO3's testing utilities.

**Files:**
- `array_test.rs` - Array creation, properties, operations
- `dtype_test.rs` - DType functionality
- `ufunc_test.rs` - Universal functions
- `iterator_test.rs` - Array iteration

**Run Rust tests:**
```bash
cargo test --lib
cargo test --lib test_array_creation  # Specific test
cargo test --lib -- --nocapture       # With output
```

### Python Pytest Tests

Located in `tests/test_*.py`. These test the public Python API.

**Files:**
- `test_array.py` - Comprehensive array tests
- `test_dtype.py` - DType tests
- `test_ufunc.py` - Ufunc tests
- `test_numpy_interop.py` - NumPy interoperability

**Run Python tests:**
```bash
# Ensure module is built first
maturin develop

# Run all tests
pytest tests/

# Run specific file
pytest tests/test_array.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=raptors
```

## Prerequisites

### Required

- Rust toolchain (`rustup`)
- Python 3.7+
- maturin (`pip install maturin` or `cargo install maturin`)
- pytest (`pip install pytest`)

### Optional

- pytest-cov for coverage (`pip install pytest-cov`)
- Make (for Makefile targets)

## Building the Module

Before running Python tests, the module must be built:

```bash
# Development build (recommended for testing)
maturin develop

# Release build (for performance testing)
maturin develop --release
```

## Running Tests

### Individual Test Suites

**Rust tests only:**
```bash
make test-rust
# or
cargo test --lib
```

**Python tests only:**
```bash
make test-python
# or
pytest tests/
```

**All tests:**
```bash
make test
# or
./run_tests.sh
```

### Test Options

**Run specific test:**
```bash
# Rust
cargo test --lib test_array_creation

# Python
pytest tests/test_array.py::TestArrayCreation::test_zeros
```

**Run with output:**
```bash
# Rust
cargo test --lib -- --nocapture

# Python
pytest tests/ -v -s
```

**Run with coverage:**
```bash
pytest tests/ --cov=raptors --cov-report=html
# Coverage report in htmlcov/
```

## Writing Tests

### Adding Rust Tests

Add to appropriate `tests/*_test.rs` file:

```rust
#[test]
fn test_new_feature() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        // Test code here
        assert!(condition);
    });
}
```

### Adding Python Tests

Add to appropriate `tests/test_*.py` file:

```python
def test_new_feature():
    """Test new feature"""
    import raptors
    # Test code here
    assert condition
```

### Test Organization

- Group related tests in classes
- Use descriptive test names
- Add docstrings explaining what is tested
- Use fixtures from `conftest.py` when appropriate

## Continuous Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'raptors'`

**Solution:**
```bash
maturin develop
```

### Import Errors in Tests

**Error:** Import errors when running pytest

**Solution:**
- Ensure `conftest.py` is in the tests directory
- Check that the module is built: `python -c "import raptors"`
- Verify PYTHONPATH if using custom setup

### Rust Tests Fail

**Error:** Rust tests fail to compile or run

**Solution:**
- Update Rust: `rustup update`
- Clean build: `cargo clean && cargo test --lib`
- Check for missing dependencies

### Python Tests Fail

**Error:** Python tests fail with import or runtime errors

**Solution:**
- Rebuild module: `maturin develop`
- Check Python version: `python --version` (needs 3.7+)
- Verify pytest is installed: `pip install pytest`
- Run with verbose output: `pytest tests/ -v -s`

## Test Coverage

Current test coverage includes:

- ✅ Array creation (zeros, ones, empty)
- ✅ Array properties (shape, size, ndim, dtype, strides)
- ✅ Array operations (arithmetic, comparison)
- ✅ Array manipulation (reshape, transpose, copy, view)
- ✅ Array indexing (getitem, setitem)
- ✅ Array iteration
- ✅ DType creation and properties
- ✅ Ufunc operations (arithmetic, math, reductions)
- ⏳ NumPy interoperability (partially implemented)

## Best Practices

1. **Always test both Rust and Python**: Rust tests verify bindings, Python tests verify API
2. **Use fixtures**: Share common setup code via `conftest.py`
3. **Test edge cases**: Empty arrays, single elements, large arrays
4. **Test error cases**: Invalid inputs, type mismatches
5. **Keep tests fast**: Use small arrays for unit tests
6. **Document tests**: Explain what each test verifies

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [PyO3 Testing Guide](https://pyo3.rs/latest/testing.html)
- [Rust Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)

