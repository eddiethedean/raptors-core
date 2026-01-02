# Building and Publishing Raptors Python Package

This document describes how to build and publish the Raptors Python package.

## Prerequisites

- Rust toolchain (install from https://rustup.rs/)
- Python 3.7 or later
- maturin (install with `pip install maturin` or `cargo install maturin`)

## Local Development Build

### Install in Editable Mode

For development, install the package in editable mode:

```bash
cd raptors-python
maturin develop
```

This will:
- Build the Rust extension
- Install it in your current Python environment
- Make changes immediately available without reinstalling

### Release Build

For optimized builds:

```bash
maturin develop --release
```

## Building Distributions

### Build Wheel

Build a wheel for your current platform:

```bash
maturin build
```

Build in release mode:

```bash
maturin build --release
```

The wheel will be created in `target/wheels/`.

### Build Source Distribution

Build a source distribution (sdist):

```bash
maturin build --sdist
```

The source distribution will be created in `target/wheels/`.

### Build for Multiple Python Versions

To build for a specific Python version:

```bash
maturin build --python 3.8
maturin build --python 3.9
maturin build --python 3.10
# etc.
```

### Build for Multiple Platforms

For cross-platform builds, use Docker or CI/CD:

```bash
# Linux (manylinux)
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

# macOS (universal)
maturin build --release --target universal2-apple-darwin

# Windows
maturin build --release
```

## Testing the Build

### Test Local Installation

1. Build the wheel:
   ```bash
   maturin build --release
   ```

2. Install from wheel:
   ```bash
   pip install target/wheels/raptors-*.whl
   ```

3. Test the installation:
   ```bash
   python -c "import raptors; print(raptors.__version__)"
   ```

### Test in Clean Environment

Create a clean virtual environment and test:

```bash
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
pip install target/wheels/raptors-*.whl
pytest tests/
deactivate
rm -rf test_env
```

## Version Management

The version must be synchronized across:

1. `Cargo.toml` (workspace root) - `version = "0.1.0"`
2. `raptors-python/Cargo.toml` - `version = "0.1.0"` (or use workspace version)
3. `raptors-python/pyproject.toml` - `version = "0.1.0"`

**Important**: When updating the version, update all three files.

## Publishing to PyPI

### Prerequisites

1. Create a PyPI account at https://pypi.org/account/register/
2. Create an API token at https://pypi.org/manage/account/token/
3. Configure credentials (see `.pypirc.example`)

### TestPyPI (Recommended First Step)

Always test on TestPyPI before publishing to PyPI:

```bash
# Build the package
maturin build --release

# Publish to TestPyPI
maturin publish --test

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ raptors
```

### PyPI Publishing

Once tested on TestPyPI:

```bash
# Build the package
maturin build --release

# Publish to PyPI
maturin publish
```

**Note**: Publishing is permanent. Make sure:
- Version number is correct
- All tests pass
- Documentation is up to date
- You've tested on TestPyPI first

### Using API Tokens

Set environment variables:

```bash
export MATURIN_PYPI_TOKEN="pypi-..."
```

Or use `.pypirc` file (see `.pypirc.example`).

## CI/CD Publishing

For automated publishing, see `.github/workflows/publish-python.yml` (if created).

The workflow should:
- Build wheels for multiple Python versions
- Build for multiple platforms (Linux, macOS, Windows)
- Publish to TestPyPI on pull requests
- Publish to PyPI on version tags

## Troubleshooting

### Build Fails

- Ensure Rust toolchain is up to date: `rustup update`
- Ensure maturin is up to date: `pip install --upgrade maturin`
- Check Python version: `python --version`
- Check for missing dependencies

### Import Errors After Installation

- Verify installation: `pip show raptors`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Reinstall: `pip uninstall raptors && maturin develop`

### Version Conflicts

- Ensure version is synced across all files
- Clear build cache: `cargo clean`
- Rebuild: `maturin build --release`

## Additional Resources

- [Maturin Documentation](https://maturin.rs/)
- [PyPI Packaging Guide](https://packaging.python.org/)
- [Python Packaging User Guide](https://packaging.python.org/guides/)

