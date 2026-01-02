# NumPy to Raptors Core Conversion Guide

This guide helps you convert NumPy code to Raptors Core.

## API Mapping

### Array Creation

**NumPy:**
```python
import numpy as np
arr = np.zeros((3, 4), dtype=np.float64)
arr = np.ones((3, 4), dtype=np.float64)
arr = np.empty((3, 4), dtype=np.float64)
```

**Raptors Core:**
```rust
use raptors_core::{zeros, ones, empty};
use raptors_core::types::{DType, NpyType};

let arr = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let arr = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let arr = empty(vec![3, 4], DType::new(NpyType::Double)).unwrap();
```

### Array Properties

**NumPy:**
```python
arr.shape
arr.dtype
arr.size
arr.ndim
arr.strides
```

**Raptors Core:**
```rust
array.shape()      // Returns &[i64]
array.dtype()      // Returns &DType
array.size()       // Returns usize
array.ndim()       // Returns usize
array.strides()    // Returns &[i64]
```

### Indexing

**NumPy:**
```python
arr[1, 2]
arr[1:3, 2:4]
arr[[0, 2], [1, 3]]  # Fancy indexing
arr[arr > 0.5]       # Boolean indexing
```

**Raptors Core:**
```rust
use raptors_core::indexing::index_array;

// Integer indexing
let indices = vec![1, 2];
let ptr = index_array(&array, &indices).unwrap();

// Slice indexing
use raptors_core::indexing::slicing::slice_array;
let slice = slice_array(&array, &[(1, 3), (2, 4)]).unwrap();

// Fancy indexing
use raptors_core::indexing::advanced::fancy::fancy_index;
let indices = vec![vec![0, 2], vec![1, 3]];
let result = fancy_index(&array, &indices).unwrap();

// Boolean indexing
use raptors_core::indexing::advanced::boolean::boolean_index;
let mask = /* create boolean array */;
let result = boolean_index(&array, &mask).unwrap();
```

### Arithmetic Operations

**NumPy:**
```python
c = a + b
c = a - b
c = a * b
c = a / b
```

**Raptors Core:**
```rust
use raptors_core::operations::{add, subtract, multiply, divide};

let c = add(&a, &b).unwrap();
let c = subtract(&a, &b).unwrap();
let c = multiply(&a, &b).unwrap();
let c = divide(&a, &b).unwrap();
```

### Shape Manipulation

**NumPy:**
```python
arr.reshape(12)
arr.T
arr.squeeze()
np.expand_dims(arr, 0)
```

**Raptors Core:**
```rust
use raptors_core::shape::{reshape, transpose, squeeze, expand_dims};

let reshaped = reshape(&array, vec![12]).unwrap();
let transposed = transpose(&array).unwrap();
let squeezed = squeeze(&array, None).unwrap();
let expanded = expand_dims(&array, 0).unwrap();
```

### Universal Functions

**NumPy:**
```python
np.sin(arr)
np.cos(arr)
np.exp(arr)
np.log(arr)
np.sqrt(arr)
```

**Raptors Core:**
```rust
use raptors_core::ufunc::loop_exec::create_unary_ufunc_loop;
use raptors_core::ufunc::advanced::math_ufuncs::*;

let sin_ufunc = create_sin_ufunc();
let mut output = empty(array.shape().to_vec(), array.dtype().clone()).unwrap();
create_unary_ufunc_loop(&sin_ufunc, &array, &mut output).unwrap();
```

### Reductions

**NumPy:**
```python
np.sum(arr, axis=0)
np.mean(arr, axis=1)
np.min(arr, axis=None)
np.max(arr, axis=None)
```

**Raptors Core:**
```rust
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis, min_along_axis, max_along_axis};

let sum = sum_along_axis(&array, Some(0)).unwrap();
let mean = mean_along_axis(&array, Some(1)).unwrap();
let min = min_along_axis(&array, None).unwrap();
let max = max_along_axis(&array, None).unwrap();
```

## Differences from NumPy

### 1. Error Handling

NumPy raises exceptions, Raptors Core returns `Result`:

**NumPy:**
```python
try:
    arr = np.zeros((3, 4))
except Exception as e:
    print(f"Error: {e}")
```

**Raptors Core:**
```rust
match zeros(vec![3, 4], dtype) {
    Ok(arr) => { /* use arr */ }
    Err(e) => { eprintln!("Error: {}", e); }
}
```

### 2. Type System

Raptors Core uses explicit types:

**NumPy:**
```python
arr = np.array([1, 2, 3])  # Inferred type
```

**Raptors Core:**
```rust
let arr = Array::from_slice(&[1, 2, 3], vec![3], DType::new(NpyType::Int)).unwrap();
```

### 3. Memory Management

Raptors Core uses Rust's ownership system:

- Arrays own their data or reference it via views
- Views use `Arc`/`Weak` for shared ownership
- No garbage collection - memory is freed when arrays go out of scope

### 4. Broadcasting

Broadcasting works similarly, but requires explicit shape computation:

**NumPy:**
```python
a = np.array([[1, 2, 3]])
b = np.array([1, 2, 3])
c = a + b  # Automatic broadcasting
```

**Raptors Core:**
```rust
use raptors_core::broadcasting::broadcast_shapes;

let broadcast_shape = broadcast_shapes(a.shape(), b.shape()).unwrap();
// Then perform operation with broadcasted shapes
```

## Common Pitfalls

### 1. Forgetting Error Handling

Always handle `Result` types:

```rust
// Wrong
let arr = zeros(shape, dtype);  // Returns Result

// Correct
let arr = zeros(shape, dtype).unwrap();  // Or use match/?
```

### 2. Type Mismatches

Ensure dtypes match:

```rust
// Wrong - type mismatch
let a = zeros(vec![3, 4], DType::new(NpyType::Int)).unwrap();
let b = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let c = add(&a, &b);  // May fail due to type promotion

// Correct - matching types or use type promotion
let a = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let b = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let c = add(&a, &b).unwrap();
```

### 3. Unsafe Pointer Access

When accessing array elements via pointers, use unsafe blocks:

```rust
let ptr = index_array(&array, &indices).unwrap();
unsafe {
    let value = *(ptr as *const f64);
    // Use value
}
```

### 4. View Lifetime

Views share memory with base arrays. Ensure base arrays outlive views:

```rust
let base = zeros(vec![3, 4], dtype).unwrap();
let view = base.view().unwrap();
// base must outlive view
```

## Performance Considerations

1. **Use Contiguous Arrays**: C-contiguous arrays are faster
2. **Avoid Unnecessary Copies**: Use views when possible
3. **Batch Operations**: Combine operations when possible
4. **Choose Appropriate Types**: Use smaller types when precision allows
5. **Parallel Operations**: Use parallel operations for large arrays

## Migration Checklist

- [ ] Replace NumPy array creation with Raptors Core equivalents
- [ ] Update error handling (exceptions → Results)
- [ ] Explicitly specify types where needed
- [ ] Handle memory ownership correctly
- [ ] Update indexing operations
- [ ] Replace NumPy ufuncs with Raptors Core equivalents
- [ ] Update shape manipulation operations
- [ ] Test thoroughly for correctness
- [ ] Benchmark performance

## Examples

See `examples/` directory for complete conversion examples.

