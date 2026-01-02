# Contributing to Raptors Core

Thank you for your interest in contributing to Raptors Core!

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Cargo
- Python 3.7+ (for Python bindings)
- Git

### Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-org/raptors.git
cd raptors
```

2. Build the project:
```bash
cargo build
```

3. Run tests:
```bash
cargo test
```

4. Run benchmarks:
```bash
cargo bench
```

## Code Style

### Rust Style

Follow Rust's official style guide:

- Use `rustfmt` for formatting:
```bash
cargo fmt
```

- Use `clippy` for linting:
```bash
cargo clippy
```

### Naming Conventions

- Functions: `snake_case`
- Types: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Modules: `snake_case`

### Documentation

- Document all public APIs
- Include examples in doc comments
- Document safety requirements for unsafe code
- Use `# Safety` sections for unsafe functions

Example:
```rust
/// Create a new array
///
/// # Example
/// ```
/// use raptors_core::{Array, DType, NpyType};
/// let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
/// ```
///
/// # Errors
/// Returns `ArrayError::AllocationFailed` if memory allocation fails.
pub fn new(shape: Vec<i64>, dtype: DType) -> Result<Self, ArrayError> {
    // ...
}
```

## Testing Requirements

### Test Coverage

- Aim for >80% test coverage per module
- Test all public APIs
- Test error cases
- Test edge cases

### Writing Tests

Tests should be in `tests/` directory:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_array_creation() {
        let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
        assert_eq!(array.shape(), &[3, 4]);
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_array_creation

# Run with output
cargo test -- --nocapture
```

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**:
```bash
git checkout -b feature/my-feature
```

3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests and linting**:
```bash
cargo test
cargo fmt
cargo clippy
```

7. **Commit your changes**:
```bash
git commit -m "Add feature X"
```

8. **Push to your fork**:
```bash
git push origin feature/my-feature
```

9. **Create a Pull Request**

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No clippy warnings
- [ ] Code formatted with rustfmt
- [ ] Tests added for new functionality
- [ ] Breaking changes documented

## Module Conversion Guidelines

When converting NumPy modules:

1. **Study NumPy's implementation** in `numpy-reference/`
2. **Match NumPy's behavior** exactly
3. **Use Rust idioms** where possible
4. **Maintain C API compatibility** if applicable
5. **Add comprehensive tests**
6. **Document differences** from NumPy

### Conversion Steps

1. Create module structure
2. Implement core functionality
3. Add tests
4. Add C API wrappers (if needed)
5. Update documentation
6. Update roadmap

## Code Review

All code must be reviewed before merging:

- At least one approval required
- All CI checks must pass
- No unresolved discussions

## Issue Reporting

When reporting issues:

1. Check existing issues first
2. Use the issue template
3. Include:
   - Rust version
   - OS and architecture
   - Minimal reproduction case
   - Expected vs actual behavior

## Questions?

- Open an issue for questions
- Check existing documentation
- Review code examples

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

