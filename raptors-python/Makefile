.PHONY: test test-rust test-python test-all build install clean help

help:
	@echo "Raptors Python Test Suite"
	@echo ""
	@echo "Available targets:"
	@echo "  make build        - Build the Python module with maturin"
	@echo "  make install      - Install the module in editable mode"
	@echo "  make test         - Run all tests (Rust + Python)"
	@echo "  make test-rust    - Run only Rust unit tests"
	@echo "  make test-python  - Run only Python pytest tests"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make help         - Show this help message"

build:
	@echo "Building Python module..."
	maturin build

install:
	@echo "Installing Python module in editable mode..."
	maturin develop

test-rust:
	@echo "Running Rust unit tests..."
	cargo test --lib

test-python:
	@echo "Running Python pytest tests..."
	@if ! python -c "import raptors" 2>/dev/null; then \
		echo "Module not found. Building first..."; \
		maturin develop; \
	fi
	pytest tests/ -v

test: test-rust test-python
	@echo "All tests completed!"

test-all: test

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf *.egg-info/
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

