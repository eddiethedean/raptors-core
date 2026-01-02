# Raptors Core

A Rust implementation of NumPy's C/C++ core, providing C API compatibility for use as a drop-in replacement.

## Overview

Raptors Core is a systematic conversion of NumPy's core C/C++ implementation to idiomatic Rust, while maintaining C API compatibility. The project aims to provide the same functionality as NumPy's core, implemented in safe Rust where possible.

## Status

**Phase 7 Complete** - Comprehensive C API compatibility layer implemented!

The project has completed 7 major phases of development:

### Core Features (Phases 1-3)
- ✅ **Array Core Structure** - Core array object with metadata, flags, and memory layout
- ✅ **Memory Management** - Memory allocation with proper alignment
- ✅ **Type System** - Dtype enumeration matching NumPy's type system
- ✅ **Array Creation** - Empty, zero-filled, and one-filled arrays
- ✅ **Indexing** - Integer indexing, slicing, fancy indexing, and boolean indexing
- ✅ **Broadcasting** - Shape computation and validation
- ✅ **Shape Manipulation** - Reshape, transpose, squeeze, expand_dims, flatten
- ✅ **Type Conversion** - Type promotion and casting safety checks
- ✅ **Iterators** - ArrayIterator, FlatIterator, StridedIterator, and advanced NdIter
- ✅ **Universal Functions** - Ufunc infrastructure with arithmetic, comparison, and advanced math functions
- ✅ **Reduction Operations** - Sum, mean, min, max with axis support
- ✅ **Array Operations** - Arithmetic and comparison operations

### Advanced Features (Phase 4)
- ✅ **Advanced Ufuncs** - Trigonometric, logarithmic, exponential, rounding functions
- ✅ **Advanced Indexing** - Fancy indexing and boolean indexing
- ✅ **Array Concatenation** - Concatenate, stack, and split operations
- ✅ **Linear Algebra** - Dot product and matrix multiplication
- ✅ **File I/O** - NPY format save/load functionality

### Extended Features (Phase 5)
- ✅ **Advanced Iterators** - Multi-array iteration (nditer) with broadcasting
- ✅ **Sorting and Searching** - Sort, argsort, searchsorted, partition operations
- ✅ **Array Manipulation** - Flip, rotate, roll, repeat, tile, unique, set operations
- ✅ **Statistical Operations** - Percentile, median, mode, std, var, correlation, histogram
- ✅ **DateTime Support** - DateTime and Timedelta dtypes with arithmetic

### Specialized Features (Phase 6)
- ✅ **String Operations** - String arrays, concatenation, comparison, formatting
- ✅ **Masked Arrays** - Masked array structure with mask propagation
- ✅ **DLPack Support** - DLPack tensor format conversion and interoperability
- ✅ **Structured Arrays** - Structured dtype with field access
- ✅ **Memory-Mapped Arrays** - Memory-mapped file arrays with lazy loading

### C API Compatibility (Phase 7)
- ✅ **Complete C API Layer** - 40+ C API wrapper functions
- ✅ **Array Views and Copies** - PyArray_View, PyArray_NewView, PyArray_Squeeze, PyArray_Flatten
- ✅ **Array Manipulation C API** - PyArray_Reshape, PyArray_Transpose, PyArray_Ravel, PyArray_SwapAxes
- ✅ **Indexing C API** - PyArray_Take, PyArray_Put, PyArray_PutMask, PyArray_Choose, PyArray_Compress
- ✅ **Concatenation C API** - PyArray_Concatenate, PyArray_Stack, PyArray_Split
- ✅ **Sorting C API** - PyArray_Sort, PyArray_ArgSort, PyArray_SearchSorted, PyArray_Partition
- ✅ **Linear Algebra C API** - PyArray_MatrixProduct, PyArray_InnerProduct, PyArray_MatMul
- ✅ **File I/O C API** - PyArray_Save, PyArray_Load
- ✅ **Operations C API** - PyArray_Broadcast, PyArray_BroadcastToShape, PyArray_Clip, PyArray_Round

## Project Structure

```
raptors-core/
├── src/
│   ├── array/          # Core array implementation
│   ├── memory/         # Memory management
│   ├── types/          # Type system (dtypes)
│   ├── indexing/       # Indexing and slicing (basic and advanced)
│   ├── broadcasting/   # Broadcasting operations
│   ├── shape/          # Shape manipulation
│   ├── conversion/     # Type conversion and promotion
│   ├── iterators/      # Array iterators (basic and advanced)
│   ├── ufunc/          # Universal functions
│   ├── operations/     # Array operations
│   ├── concatenation/  # Concatenation and splitting
│   ├── linalg/         # Linear algebra operations
│   ├── io/             # File I/O (NPY format)
│   ├── sorting/        # Sorting and searching
│   ├── manipulation/   # Array manipulation utilities
│   ├── statistics/     # Statistical operations
│   ├── datetime/       # DateTime and Timedelta support
│   ├── string/         # String array operations
│   ├── masked/         # Masked array support
│   ├── structured/     # Structured arrays
│   ├── dlpack/         # DLPack support
│   ├── memmap/         # Memory-mapped arrays
│   ├── ffi/            # C API compatibility layer
│   │   ├── array_api.rs      # Array property and creation functions
│   │   ├── conversion.rs     # Array <-> PyArrayObject conversion
│   │   ├── views.rs          # Array views and copies
│   │   ├── manipulation.rs   # Array manipulation C API
│   │   ├── indexing.rs       # Indexing C API
│   │   ├── concatenation.rs    # Concatenation C API
│   │   ├── sorting.rs        # Sorting C API
│   │   ├── linalg.rs         # Linear algebra C API
│   │   ├── io.rs             # File I/O C API
│   │   └── operations.rs     # Advanced operations C API
│   └── utils/          # Utilities
├── tests/              # Comprehensive test suite (180+ tests)
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

**Test Coverage:**
- 180+ unit tests across 24 test files
- Comprehensive coverage of all implemented modules
- C API integration tests
- Test coverage includes:
  - Array creation and properties (8 tests)
  - Indexing - basic and advanced (9 tests)
  - Slicing (6 tests)
  - Broadcasting (8 tests)
  - Shape operations (11 tests)
  - Ufuncs - advanced (8 tests)
  - Reductions (8 tests)
  - Array operations (7 tests)
  - Iterators - basic and advanced (9 tests)
  - Concatenation (4 tests)
  - Linear algebra (3 tests)
  - File I/O (2 tests)
  - FFI/C API (30 tests)
  - Sorting and searching (6 tests)
  - Array manipulation (10 tests)
  - Statistical operations (8 tests)
  - DateTime operations (7 tests)
  - String operations (21 tests)
  - Masked arrays (17 tests)
  - Structured arrays (11 tests)
  - DLPack support (8 tests)
  - Memory-mapped arrays (6 tests)

## C API Compatibility

The crate provides comprehensive C API compatibility through the `ffi` module. C headers are generated using `cbindgen` and placed in `target/include/raptors_core.h`.

### Implemented C API Functions (40+ functions)

**Array Properties:**
- `PyArray_SIZE`, `PyArray_NDIM`, `PyArray_DIM`, `PyArray_STRIDE`
- `PyArray_DATA`, `PyArray_DIMS`, `PyArray_STRIDES`, `PyArray_ITEMSIZE`

**Array Creation:**
- `PyArray_New`, `PyArray_NewFromDescr`, `PyArray_Empty`, `PyArray_Zeros`, `PyArray_Ones`

**Type Checking:**
- `PyArray_Check`, `PyArray_CheckExact`

**Array Views and Copies:**
- `PyArray_View`, `PyArray_NewView`, `PyArray_Squeeze`, `PyArray_Flatten`

**Array Manipulation:**
- `PyArray_Reshape`, `PyArray_Transpose`, `PyArray_Ravel`, `PyArray_SwapAxes`

**Indexing and Selection:**
- `PyArray_Take`, `PyArray_Put`, `PyArray_PutMask`, `PyArray_Choose`, `PyArray_Compress`

**Concatenation and Splitting:**
- `PyArray_Concatenate`, `PyArray_Stack`, `PyArray_Split`

**Sorting and Searching:**
- `PyArray_Sort`, `PyArray_ArgSort`, `PyArray_SearchSorted`, `PyArray_Partition`

**Linear Algebra:**
- `PyArray_MatrixProduct`, `PyArray_InnerProduct`, `PyArray_MatMul`

**File I/O:**
- `PyArray_Save`, `PyArray_Load`

**Advanced Operations:**
- `PyArray_Broadcast`, `PyArray_BroadcastToShape`, `PyArray_Clip`, `PyArray_Round`

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

**Completed Phases:** 1-7 (Core functionality, Advanced features, Extended features, Specialized features, and C API compatibility)

**Future Enhancements:**
- Performance optimizations (SIMD, parallel processing)
- Additional NumPy C API functions
- Python bindings (via PyO3)
- GPU array support
- Custom dtype creation API
- Enhanced memory layout optimizations
- Comprehensive documentation and examples

## License

This project is a reimplementation of NumPy's core functionality. Please refer to NumPy's license for compatibility requirements.

## References

- [NumPy Repository](https://github.com/numpy/numpy)
- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy C API Documentation](https://numpy.org/doc/stable/reference/c-api/)

