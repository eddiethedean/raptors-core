# Raptors Core Performance Guide

This guide covers performance characteristics and optimization tips for Raptors Core.

## Performance Characteristics

### Array Creation

- **Empty arrays**: Fast allocation, uninitialized memory
- **Zero-filled arrays**: Requires memory initialization (slower)
- **One-filled arrays**: Requires type-specific initialization (slower)

**Tip**: Use `empty()` when you'll fill the array yourself.

### Memory Layout

- **C-contiguous arrays**: Fastest for most operations
- **F-contiguous arrays**: Fastest for column-major operations
- **Strided arrays**: Slower due to non-contiguous memory access

**Tip**: Use `as_contiguous()` to ensure optimal layout.

### Operations

- **Contiguous arrays**: Fast paths for contiguous memory
- **Strided arrays**: Slower due to stride calculations
- **Large arrays**: Benefit from parallel operations

## Optimization Tips

### 1. Use Contiguous Arrays

```rust
// Ensure array is contiguous
let array = array.as_contiguous(Order::C).unwrap();
```

### 2. Avoid Unnecessary Copies

```rust
// Use views instead of copies when possible
let view = array.view().unwrap();
// Instead of
let copy = array.copy().unwrap();
```

### 3. Batch Operations

```rust
// Combine operations
let result = add(&a, &b)
    .and_then(|sum| multiply(&sum, &c))
    .unwrap();
```

### 4. Choose Appropriate Types

```rust
// Use smaller types when precision allows
let array = zeros(shape, DType::new(NpyType::Float));  // 32-bit
// Instead of
let array = zeros(shape, DType::new(NpyType::Double));  // 64-bit
```

### 5. Use Parallel Operations

For large arrays, parallel operations are automatically used:

```rust
// Automatic parallelization for large arrays
let sum = sum_along_axis(&large_array, None).unwrap();
```

## Memory Access Patterns

### Cache-Friendly Access

- Access elements in order (C-contiguous)
- Process data in blocks
- Minimize random access

### Example: Block Processing

```rust
// Process array in blocks for cache efficiency
let block_size = 1024;
for i in (0..array.size()).step_by(block_size) {
    let end = (i + block_size).min(array.size());
    // Process block [i..end]
}
```

## Threading

### Parallel Operations

Raptors Core uses Rayon for parallel operations:

- Automatic threshold detection
- Configurable thread pool
- Thread-safe operations

### Thread Pool Configuration

Set `RAYON_NUM_THREADS` environment variable:

```bash
export RAYON_NUM_THREADS=4
```

## Benchmarking

Use the benchmark suite to measure performance:

```bash
cargo bench
```

### Benchmark Categories

1. **Array Creation**: `benches/array_creation.rs`
2. **Operations**: `benches/operations.rs`
3. **Indexing**: `benches/indexing.rs`
4. **NumPy Comparison**: `benches/numpy_comparison.rs`

## Performance vs NumPy

Raptors Core aims to match NumPy's performance:

- **Array creation**: Similar performance
- **Operations**: Similar or better for contiguous arrays
- **Indexing**: Similar performance
- **Reductions**: Similar or better with parallel operations

See benchmark results in `benches/` for detailed comparisons.

## Profiling

Use Rust profiling tools:

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile your code
cargo flamegraph --bench array_creation
```

## Common Performance Issues

### 1. Non-Contiguous Arrays

**Problem**: Strided arrays are slower

**Solution**: Use `as_contiguous()` when possible

### 2. Unnecessary Copies

**Problem**: Copying large arrays is expensive

**Solution**: Use views when possible

### 3. Type Conversions

**Problem**: Type conversions add overhead

**Solution**: Use consistent types throughout

### 4. Small Array Overhead

**Problem**: Parallel operations have overhead for small arrays

**Solution**: Operations automatically detect array size and use sequential for small arrays

## Best Practices

1. **Profile First**: Measure before optimizing
2. **Use Appropriate Types**: Match precision needs
3. **Minimize Copies**: Use views and references
4. **Batch Operations**: Combine when possible
5. **Use Parallel Operations**: For large arrays
6. **Ensure Contiguity**: For hot paths

## Future Optimizations

Planned optimizations:

- Advanced SIMD support
- GPU array support
- JIT compilation
- Advanced cache optimizations
- Memory pool management

