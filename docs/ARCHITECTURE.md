# Raptors Core Architecture

This document describes the architecture and design decisions of Raptors Core.

## Repository Structure

Raptors Core uses a Rust workspace structure with two main crates:

- **`raptors-core/`**: The core Rust library implementing NumPy's functionality
- **`raptors-python/`**: Python bindings using PyO3

This structure allows both crates to be developed together while maintaining clear separation of concerns.

## Overview

Raptors Core is a Rust implementation of NumPy's C/C++ core, providing C API compatibility for use as a drop-in replacement. The project aims to match NumPy's functionality while leveraging Rust's safety guarantees.

## Core Design Principles

1. **NumPy Compatibility**: Maintain compatibility with NumPy's C API and behavior
2. **Memory Safety**: Use Rust's type system to ensure memory safety where possible
3. **Performance**: Match or exceed NumPy's performance characteristics
4. **Idiomatic Rust**: Use Rust idioms while maintaining compatibility

## Architecture Components

### 1. Array Core (`src/array/`)

The core array structure is defined in `arrayobject.rs`. Key components:

- **Array Structure**: Core array object with data pointer, shape, strides, dtype, and flags
- **Memory Management**: Automatic memory management with proper alignment
- **Views**: Zero-copy views using `Arc` and `Weak` references
- **Flags**: Array flags (C-contiguous, F-contiguous, writeable, etc.)

**Key Design Decisions**:
- Uses raw pointers for data storage (required for C API compatibility)
- Manages memory ownership with `owns_data` flag
- Uses `Arc<Array>` for shared ownership in views
- Uses `Weak<Array>` to prevent circular references

### 2. Type System (`src/types/`)

The type system matches NumPy's dtype system:

- **NpyType**: Enumeration of NumPy-compatible types
- **DType**: Type descriptor with metadata (itemsize, alignment, name)
- **Type Promotion**: Automatic type promotion in operations
- **Type Casting**: Safe type casting with validation

### 3. Memory Management (`src/memory/`)

Memory allocation with proper alignment:

- Uses Rust's `std::alloc` for memory allocation
- Respects dtype alignment requirements
- Handles large arrays (>2GB) through proper size calculations
- Memory-mapped arrays via `memmap2` crate

### 4. Broadcasting (`src/broadcasting/`)

Shape computation and validation for broadcasting:

- Computes broadcast shapes from input shapes
- Validates broadcasting compatibility
- Calculates broadcast strides
- Supports NumPy's broadcasting rules

### 5. Universal Functions (`src/ufunc/`)

Ufunc infrastructure for element-wise operations:

- **Ufunc Structure**: Generic ufunc with type resolution
- **Loop Execution**: Efficient loop execution with type dispatch
- **Parallel Execution**: Parallel ufuncs using Rayon
- **Reductions**: Sum, mean, min, max with axis support

**Architecture**:
- Type resolution: Determines output types from input types
- Loop registration: Registers type-specific loop functions
- Execution: Dispatches to appropriate loop based on types

### 6. Iterators (`src/iterators/`)

Efficient array iteration:

- **ArrayIterator**: Multi-dimensional iteration with coordinate tracking
- **FlatIterator**: Flat iteration over all elements
- **StridedIterator**: Iteration with custom strides
- **NdIter**: Multi-array iteration with broadcasting

### 7. Indexing (`src/indexing/`)

Array indexing and slicing:

- **Integer Indexing**: Direct element access
- **Slice Indexing**: Slice-based access with normalization
- **Fancy Indexing**: Integer array indexing
- **Boolean Indexing**: Mask-based indexing

### 8. Shape Manipulation (`src/shape/`)

Array shape operations:

- **Reshape**: Change array shape (with validation)
- **Transpose**: Transpose array dimensions
- **Squeeze**: Remove dimensions of size 1
- **Expand Dims**: Add dimensions of size 1

### 9. Operations (`src/operations/`)

High-level array operations:

- **Arithmetic**: Add, subtract, multiply, divide
- **Comparison**: Equal, less, greater, etc.
- Built on ufunc infrastructure

### 10. C API Compatibility (`src/ffi/`)

C API wrapper layer:

- **PyArrayObject**: C-compatible array structure
- **C API Functions**: 40+ NumPy-compatible C functions
- **Conversion**: Array <-> PyArrayObject conversion
- **Memory Management**: Proper memory handling for C API

## Memory Layout

Arrays support two memory layouts:

1. **C-contiguous (row-major)**: Last dimension stride = itemsize
2. **Fortran-contiguous (column-major)**: First dimension stride = itemsize

Strides are computed automatically based on shape and itemsize.

## Threading

Threading support via Rayon:

- Parallel reductions for large arrays
- Parallel ufunc operations
- Configurable thread pool
- Automatic threshold detection (only parallelize for large arrays)

## Error Handling

Error types:

- **ArrayError**: Array-specific errors (allocation, shape, type mismatch)
- **BroadcastError**: Broadcasting errors
- **UfuncError**: Ufunc errors (unsupported types, invalid inputs)
- **LoopExecutionError**: Loop execution errors

All errors implement `std::error::Error` for compatibility.

## Safety Considerations

### Unsafe Code

Unsafe code is used for:

1. **Raw Pointer Operations**: Required for C API compatibility
2. **Memory Allocation**: Direct allocation with custom layouts
3. **Type Casting**: Converting between types at runtime
4. **FFI**: C API compatibility layer

All unsafe code is:
- Documented with safety requirements
- Minimized to necessary operations
- Validated with tests

### Memory Safety

- Arrays own their data or reference it via views
- Views use `Arc`/`Weak` to prevent use-after-free
- Proper cleanup in `Drop` implementation
- No data races (immutable by default, mutable only when writeable)

## Performance Optimizations

1. **Contiguous Fast Paths**: Optimized paths for contiguous arrays
2. **Pairwise Summation**: Accurate summation for large arrays
3. **Cache-Friendly Algorithms**: Blocked operations for cache efficiency
4. **Parallel Operations**: Multi-threaded operations for large arrays
5. **Copy Avoidance**: Views and zero-copy operations where possible

## Future Enhancements

- Advanced SIMD optimizations
- GPU array support
- JIT compilation
- Async support
- Advanced memory layout optimizations

