# Raptors Core API Guide

This guide provides examples and patterns for using Raptors Core.

## Getting Started

### Basic Array Creation

```rust
use raptors_core::{Array, zeros, ones, empty};
use raptors_core::types::{DType, NpyType};

// Create a zero-filled array
let shape = vec![3, 4];
let dtype = DType::new(NpyType::Double);
let array = zeros(shape.clone(), dtype).unwrap();

// Create a one-filled array
let array = ones(shape.clone(), dtype.clone()).unwrap();

// Create an empty (uninitialized) array
let array = empty(shape, dtype).unwrap();
```

### Using the Builder Pattern

```rust
use raptors_core::array::builder::ArrayBuilder;
use raptors_core::types::{DType, NpyType};

let array = ArrayBuilder::new()
    .with_shape(vec![3, 4])
    .with_dtype(DType::new(NpyType::Double))
    .with_fill_value(5.0)
    .build()
    .unwrap();
```

## Array Properties

```rust
// Get array properties
println!("Shape: {:?}", array.shape());
println!("Size: {}", array.size());
println!("NDim: {}", array.ndim());
println!("DType: {:?}", array.dtype());
println!("Is C-contiguous: {}", array.is_c_contiguous());
println!("Is writeable: {}", array.is_writeable());
```

## Indexing

### Integer Indexing

```rust
use raptors_core::indexing::index_array;

let indices = vec![1, 2];
let element_ptr = index_array(&array, &indices).unwrap();

// Access value (unsafe - must match dtype)
unsafe {
    let value = *(element_ptr as *const f64);
    println!("Value at [1, 2]: {}", value);
}
```

### Slice Indexing

```rust
use raptors_core::indexing::slicing::slice_array;

let slice = slice_array(&array, &[(0, 2), (1, 3)]).unwrap();
```

## Array Operations

### Arithmetic Operations

```rust
use raptors_core::operations::{add, subtract, multiply, divide};

let a = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let b = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();

// Element-wise operations
let sum = add(&a, &b).unwrap();
let diff = subtract(&a, &b).unwrap();
let prod = multiply(&a, &b).unwrap();
let quot = divide(&a, &b).unwrap();
```

### Comparison Operations

```rust
use raptors_core::operations::{equal, less, greater};

let a = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let b = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();

let eq = equal(&a, &b).unwrap();
let lt = less(&a, &b).unwrap();
let gt = greater(&a, &b).unwrap();
```

## Shape Manipulation

```rust
use raptors_core::shape::{reshape, transpose, squeeze, expand_dims};

// Reshape array
let reshaped = reshape(&array, vec![12]).unwrap();

// Transpose array
let transposed = transpose(&array).unwrap();

// Remove dimensions of size 1
let squeezed = squeeze(&array, None).unwrap();

// Add dimensions
let expanded = expand_dims(&array, 0).unwrap();
```

## Universal Functions

### Mathematical Functions

```rust
use raptors_core::ufunc::loop_exec::create_unary_ufunc_loop;
use raptors_core::ufunc::advanced::math_ufuncs::*;

let array = zeros(vec![10], DType::new(NpyType::Double)).unwrap();

// Create ufunc
let sin_ufunc = create_sin_ufunc();
let mut output = empty(vec![10], DType::new(NpyType::Double)).unwrap();

// Execute ufunc
create_unary_ufunc_loop(&sin_ufunc, &array, &mut output).unwrap();
```

### Reductions

```rust
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis};

let array = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();

// Sum along axis 0
let sum = sum_along_axis(&array, Some(0)).unwrap();

// Mean along axis 1
let mean = mean_along_axis(&array, Some(1)).unwrap();

// Sum all elements
let total = sum_along_axis(&array, None).unwrap();
```

## Iteration

### Flat Iteration

```rust
use raptors_core::iterators::FlatIterator;

let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let mut iter = FlatIterator::new(&array);

while let Some(ptr) = iter.next() {
    // Process element at ptr
    unsafe {
        let value = *(ptr as *const f64);
        println!("Value: {}", value);
    }
}
```

### Multi-dimensional Iteration

```rust
use raptors_core::iterators::ArrayIterator;

let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let mut iter = ArrayIterator::new(&array);

while let Some(ptr) = iter.next() {
    let coords = iter.coordinates();
    println!("Element at {:?}: {:p}", coords, ptr);
}
```

## Broadcasting

```rust
use raptors_core::broadcasting::broadcast_shapes;

let shape1 = vec![3, 4];
let shape2 = vec![4];

let broadcast_shape = broadcast_shapes(&shape1, &shape2).unwrap();
println!("Broadcast shape: {:?}", broadcast_shape);
```

## Type Conversion

```rust
use raptors_core::conversion::{promote_dtypes, cast_array};

let dtype1 = DType::new(NpyType::Int);
let dtype2 = DType::new(NpyType::Double);

// Promote types
let promoted = promote_dtypes(&dtype1, &dtype2).unwrap();

// Cast array
let array = zeros(vec![3, 4], dtype1).unwrap();
let casted = cast_array(&array, &dtype2).unwrap();
```

## Convenience Methods

### From Slice

```rust
let data = vec![1.0, 2.0, 3.0, 4.0];
let array = Array::from_slice(&data, vec![2, 2], DType::new(NpyType::Double)).unwrap();
```

### To Vec

```rust
let array = ones(vec![4], DType::new(NpyType::Double)).unwrap();
let vec: Vec<f64> = unsafe { array.to_vec().unwrap() };
```

### Iterator

```rust
let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
for ptr in array.iter() {
    // Process element
}
```

## Performance Tips

1. **Use Contiguous Arrays**: C-contiguous arrays are faster for most operations
2. **Avoid Unnecessary Copies**: Use views when possible
3. **Use Parallel Operations**: Large arrays benefit from parallel operations
4. **Choose Appropriate Types**: Use smaller types when precision allows
5. **Batch Operations**: Combine multiple operations when possible

## Error Handling

All operations return `Result` types:

```rust
match zeros(shape, dtype) {
    Ok(array) => {
        // Use array
    }
    Err(e) => {
        eprintln!("Error creating array: {}", e);
    }
}
```

## Common Patterns

### Creating and Filling Arrays

```rust
let mut array = empty(vec![100], DType::new(NpyType::Double)).unwrap();
unsafe {
    array.fill_typed(42.0).unwrap();
}
```

### Working with Views

```rust
let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let view = array.view().unwrap();
// View shares memory with array
```

### Combining Operations

```rust
let a = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let b = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
let result = add(&a, &b)
    .and_then(|sum| multiply(&sum, &a))
    .unwrap();
```

