# Raptors Core

A Rust implementation of NumPy's C/C++ core, providing C API compatibility for use as a drop-in replacement.

## Overview

Raptors Core is a systematic conversion of NumPy's core C/C++ implementation to idiomatic Rust, while maintaining C API compatibility. The project aims to provide the same functionality as NumPy's core, implemented in safe Rust where possible.

## Status

This is an initial implementation with core functionality:

- ✅ **Array Core Structure** - Core array object with metadata, flags, and memory layout
- ✅ **Memory Management** - Basic memory allocation with proper alignment
- ✅ **Type System** - Dtype enumeration and properties matching NumPy's type system
- ✅ **Array Creation** - Functions for creating empty, zero-filled, and one-filled arrays
- ✅ **Basic Indexing** - Integer indexing with bounds checking
- ✅ **C API Infrastructure** - FFI layer setup with cbindgen for C header generation
- ✅ **Testing Framework** - Test infrastructure with unit and integration tests

## Project Structure

```
raptors-core/
├── src/
│   ├── array/          # Core array implementation
│   │   ├── arrayobject.rs  # Main array structure
│   │   ├── flags.rs        # Array flags
│   │   └── creation.rs     # Array creation functions
│   ├── memory/         # Memory management
│   ├── types/          # Type system (dtypes)
│   ├── indexing/       # Indexing and slicing
│   ├── ffi/            # C API compatibility layer
│   ├── ufunc/          # Universal functions (placeholder)
│   └── utils/          # Utilities
├── tests/              # Integration tests
├── docs/               # Documentation
│   └── CONVERSION_ROADMAP.md  # Conversion tracking
└── numpy-reference/    # NumPy repository for reference
```

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## C API Compatibility

The crate provides C API compatibility through the `ffi` module. C headers are generated using `cbindgen` and placed in `target/include/raptors_core.h`.

Currently implemented C API functions:
- `PyArray_SIZE` - Get array size
- `PyArray_Check` - Check if object is an array (placeholder)
- `PyArray_New` - Array creation (placeholder)

## Usage Example

```rust
use raptors_core::{Array, zeros, ones, empty};
use raptors_core::types::{DType, NpyType};
use raptors_core::indexing::index_array;

// Create a zero-filled array
let shape = vec![3, 4];
let dtype = DType::new(NpyType::Double);
let array = zeros(shape.clone(), dtype).unwrap();

// Access array properties
println!("Shape: {:?}", array.shape());
println!("Size: {}", array.size());
println!("Is C-contiguous: {}", array.is_c_contiguous());

// Index into the array
let indices = vec![1, 2];
let element_ptr = index_array(&array, &indices).unwrap();
```

## Next Steps

See `docs/CONVERSION_ROADMAP.md` for the detailed conversion plan and progress tracking.

Key areas for future implementation:
- Universal functions (ufuncs)
- Broadcasting
- Advanced indexing (slicing, fancy indexing)
- Array operations (arithmetic, comparison)
- Linear algebra operations
- Full C API implementation

## License

This project is a reimplementation of NumPy's core functionality. Please refer to NumPy's license for compatibility requirements.

## References

- [NumPy Repository](https://github.com/numpy/numpy)
- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy C API Documentation](https://numpy.org/doc/stable/reference/c-api/)

