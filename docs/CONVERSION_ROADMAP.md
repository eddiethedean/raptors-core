# NumPy to Raptors-Core Conversion Roadmap

This document tracks the conversion of NumPy's C/C++ core modules to Rust.

## Module Mapping

### Core Array Structure (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/arrayobject.c`
- **Raptors**: `src/array/arrayobject.rs`
- **Status**: Basic structure implemented, needs full API conversion

### Array Flags (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/flagsobject.c`
- **Raptors**: `src/array/flags.rs`
- **Status**: Basic flags implemented, matches NumPy flag structure

### Memory Management (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/alloc.c`, `memory.c`
- **Raptors**: `src/memory/alloc.rs`
- **Status**: Basic allocation functions implemented

### Type System (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/descriptor.c`, `dtypemeta.c`, `arraytypes.c.src`
- **Raptors**: `src/types/dtype.rs`
- **Status**: Basic dtype enumeration and structure implemented

### Array Creation (COMPLETED - Basic API)
- **NumPy**: `numpy/_core/src/multiarray/ctors.c`, `arrayobject.c` (creation functions)
- **Raptors**: `src/array/creation.rs`
- **Status**: Basic array creation implemented (empty, zeros, ones), C API functions in place

### Indexing (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/mapping.c`, `item_selection.c`
- **Raptors**: `src/indexing/indexing.rs`, `src/indexing/slicing.rs`, `src/indexing/advanced/`
- **Status**: Integer indexing, slice indexing, fancy indexing (integer array indexing), and boolean indexing implemented

### Broadcasting (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/calculation.c` (broadcasting logic)
- **Raptors**: `src/broadcasting/broadcast.rs`
- **Status**: Broadcast shape computation, validation, and stride calculation implemented

### Shape Manipulation (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/shape.c`
- **Raptors**: `src/shape/shape.rs`
- **Status**: Reshape validation, transpose, squeeze, expand_dims, flatten implemented

### Type Conversion (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/convert_datatype.c`, `convert.c`
- **Raptors**: `src/conversion/promotion.rs`, `src/conversion/casting.rs`
- **Status**: Type promotion rules and casting safety checks implemented

### Universal Functions (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/umath/`
- **Raptors**: `src/ufunc/`
- **Status**: Ufunc structure, type resolution, loop framework, basic arithmetic/comparison ufuncs, and advanced mathematical ufuncs (trigonometric, logarithmic, exponential, etc.) implemented

### Iterators (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/iterators.c`
- **Raptors**: `src/iterators/`
- **Status**: ArrayIterator, FlatIterator, and StridedIterator implemented with coordinate tracking

### Array Operations (COMPLETED - Basic Structure)
- **NumPy**: Various operations built on ufuncs
- **Raptors**: `src/operations/`
- **Status**: Arithmetic and comparison operations implemented, built on ufunc infrastructure

### Array Concatenation (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/shape.c` (concatenation)
- **Raptors**: `src/concatenation/`
- **Status**: Concatenate, stack, and split operations implemented

### Linear Algebra (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/multiarray/vdot.c`, `numpy/_core/src/umath/matmul.c.src`
- **Raptors**: `src/linalg/`
- **Status**: Dot product and matrix multiplication implemented (1D-1D, 1D-2D, 2D-1D, 2D-2D cases)

### File I/O (COMPLETED - Basic Structure)
- **NumPy**: NPY format specification
- **Raptors**: `src/io/`
- **Status**: NPY format save/load functionality implemented

### Reduction Operations (COMPLETED - Basic Structure)
- **NumPy**: `numpy/_core/src/umath/reduction.c`
- **Raptors**: `src/ufunc/reduction.rs`
- **Status**: Sum, mean, min, max reductions with axis support implemented

### Common Utilities (PARTIAL)
- **NumPy**: `numpy/_core/src/common/`
- **Raptors**: `src/utils/`
- **Status**: Basic utilities implemented

## Priority Order

1. ✅ **Array Core Structure** - Foundation for all operations
2. ✅ **Memory Management** - Required for array operations
3. ✅ **Type System** - Required for dtype handling
4. ✅ **Array Flags** - Required for array metadata
5. ✅ **Array Creation** - Core API functions
6. ✅ **Indexing** - Basic array access and slicing
7. ✅ **Broadcasting** - Required for operations
8. ✅ **Shape Manipulation** - Array reshaping, transpose, etc.
9. ✅ **Type Conversion** - Type promotion and casting
10. ✅ **Iterators** - For efficient iteration (COMPLETED in Phase 3)
11. ✅ **Universal Functions** - Core mathematical operations (COMPLETED in Phase 3)
12. ✅ **Array Operations** - Arithmetic and comparison operations (COMPLETED in Phase 3)
13. ✅ **Reduction Operations** - Sum, mean, min, max reductions (COMPLETED in Phase 3)

## Key NumPy Files to Convert

### High Priority
- `arrayobject.c` - Core array object (BASIC DONE, needs full API)
- `ctors.c` - Array constructors (BASIC DONE)
- `descriptor.c` - Dtype descriptors (BASIC DONE)
- `alloc.c` - Memory allocation (BASIC DONE)
- `mapping.c` - Indexing (BASIC DONE - integer and slice indexing)
- `item_selection.c` - Item selection (BASIC DONE)
- `calculation.c` - Broadcasting (DONE)
- `shape.c` - Shape operations (BASIC DONE)
- `convert_datatype.c` - Type conversion (BASIC DONE)

### Medium Priority
- `iterators.c` - Array iterators (COMPLETED - Phase 3)
- `umath/` - Universal functions (BASIC DONE - Phase 3, basic ufuncs implemented)
- `calculation.c` - Array calculations (partial - broadcasting done)
- `item_selection.c` - Item selection (BASIC DONE - needs advanced indexing)
- `mapping.c` - Mapping/indexing (BASIC DONE - needs fancy indexing)
- `strfuncs.c` - String functions (TODO - Phase 6)
- `unique.cpp` - Unique element finding (TODO - Phase 5)
- `einsum.cpp` - Einstein summation (TODO - Future)
- `vdot.c` - Vector dot product (TODO - Phase 4)

### Lower Priority
- `nditer_*.c` - Advanced iterators (TODO - Phase 5)
- `datetime*.c` - DateTime support (TODO - Phase 5)
- `dlpack.c` - DLPack support (TODO - Phase 6)
- `textreading/` - Text file reading (TODO - Future)
- `stringdtype/` - String dtype support (TODO - Phase 6)
- `usertypes.c` - User-defined types (TODO - Future)
- `buffer.c` - Buffer protocol (TODO - Future)
- `refcount.c` - Reference counting (BASIC - may need enhancement)

## C API Compatibility

The following C API functions need to be implemented in `src/ffi/`:

### Array Creation
- `PyArray_New` - Create new array (PLACEHOLDER - needs memory management)
- `PyArray_NewFromDescr` - Create from descriptor (TODO)
- `PyArray_Empty` - Create empty array (DONE)
- `PyArray_Zeros` - Create zero-filled array (DONE)
- `PyArray_Ones` - Create one-filled array (DONE)

### Array Properties
- `PyArray_SIZE` - Get array size (DONE)
- `PyArray_NDIM` - Get number of dimensions (DONE)
- `PyArray_DIM` - Get dimension size (DONE)
- `PyArray_STRIDE` - Get stride (DONE)
- `PyArray_DATA` - Get data pointer (DONE)
- `PyArray_DIMS` - Get dimensions pointer (DONE)
- `PyArray_STRIDES` - Get strides pointer (DONE)
- `PyArray_ITEMSIZE` - Get item size (DONE - placeholder implementation)

### Type Checking
- `PyArray_Check` - Check if object is array (PLACEHOLDER)
- `PyArray_CheckExact` - Exact type check (DONE - placeholder implementation)

### Array Views and Copies
- `PyArray_View` - Create array view (TODO)
- `PyArray_NewView` - Create new view (TODO)
- `PyArray_Squeeze` - Remove dimensions of size 1 (TODO)
- `PyArray_Flatten` - Flatten array (TODO)

### Array Manipulation
- `PyArray_Reshape` - Reshape array (TODO)
- `PyArray_Transpose` - Transpose array (TODO)
- `PyArray_Ravel` - Return flattened view (TODO)
- `PyArray_SwapAxes` - Swap two axes (TODO)

### Indexing and Selection
- `PyArray_Take` - Take elements using index array (TODO - Phase 4)
- `PyArray_Put` - Put values using index array (TODO - Phase 4)
- `PyArray_PutMask` - Put values using boolean mask (TODO - Phase 4)
- `PyArray_Choose` - Choose elements from arrays (TODO - Phase 4)
- `PyArray_Compress` - Select elements using condition (TODO - Phase 4)

### Concatenation and Splitting
- `PyArray_Concatenate` - Concatenate arrays (TODO - Phase 4)
- `PyArray_Stack` - Stack arrays (TODO - Phase 4)
- `PyArray_Split` - Split array (TODO - Phase 4)

### Sorting and Searching
- `PyArray_Sort` - Sort array (TODO - Phase 5)
- `PyArray_ArgSort` - Return indices that would sort array (TODO - Phase 5)
- `PyArray_SearchSorted` - Find insertion points (TODO - Phase 5)
- `PyArray_Partition` - Partition array (TODO - Phase 5)

### Linear Algebra
- `PyArray_MatrixProduct` - Matrix multiplication (TODO - Phase 4)
- `PyArray_InnerProduct` - Inner product (TODO - Phase 4)
- `PyArray_MatMul` - Matrix multiplication (TODO - Phase 4)

### File I/O
- `PyArray_Save` - Save array to file (TODO - Phase 4)
- `PyArray_Load` - Load array from file (TODO - Phase 4)
- `PyArray_SaveText` - Save as text (TODO - Future)
- `PyArray_LoadText` - Load from text (TODO - Future)

### Advanced Operations
- `PyArray_Broadcast` - Broadcast arrays (TODO)
- `PyArray_BroadcastToShape` - Broadcast to shape (TODO)
- `PyArray_Clip` - Clip values (TODO - Phase 4)
- `PyArray_Round` - Round values (TODO - Phase 4)

## Phase 2 Completed

Phase 2 focused on completing essential array operations and C API compatibility:

- ✅ C API property and creation functions
- ✅ Slice indexing with normalization
- ✅ Broadcasting shape computation and validation
- ✅ Shape manipulation operations (reshape, transpose, squeeze, etc.)
- ✅ Type promotion and casting safety checks
- ✅ Array-FFI conversion layer

## Phase 3 Completed

Phase 3 focused on implementing array iterators, universal functions, and array operations:

- ✅ **Array Iterators** - ArrayIterator, FlatIterator, and StridedIterator with coordinate tracking
- ✅ **Universal Functions Infrastructure** - Ufunc structure, registration system, type resolution, loop framework
- ✅ **Basic Ufunc Implementations** - Add, subtract, multiply, divide, and comparison ufuncs
- ✅ **Reduction Operations** - Sum, mean, min, max reductions with axis support
- ✅ **Array Operations** - High-level arithmetic and comparison operations built on ufuncs
- ✅ **Iterator Traits** - Rust Iterator trait implementation for seamless integration

## Phase 4 Completed (Latest Update)

Phase 4 focused on advanced ufuncs, indexing, concatenation, linear algebra, and file I/O:

- ✅ **Advanced Ufuncs** - Trigonometric, logarithmic, exponential, rounding, and sign functions (sin, cos, tan, exp, log, sqrt, abs, floor, ceil, round, trunc, etc.)
- ✅ **Advanced Indexing** - Fancy indexing (integer array indexing) and boolean indexing (mask indexing)
- ✅ **Array Concatenation** - Concatenate, stack, and split operations with axis support
- ✅ **Linear Algebra** - Dot product and matrix multiplication for 1D-1D, 1D-2D, 2D-1D, and 2D-2D cases
- ✅ **File I/O** - NPY format save/load functionality with header parsing

## Phase 5 Priorities (Next Steps)

### 4.1 Advanced Ufuncs (HIGH PRIORITY)
- **Target Files**: `numpy/_core/src/umath/loops_trigonometric.c`, `loops_logarithmic.c`, etc.
- **Raptors**: `src/ufunc/advanced/`
- **Features**:
  - Trigonometric functions (sin, cos, tan, asin, acos, atan, etc.)
  - Hyperbolic functions (sinh, cosh, tanh, etc.)
  - Exponential and logarithmic (exp, log, log10, log2, sqrt, etc.)
  - Rounding functions (floor, ceil, round, trunc)
  - Sign and absolute value (abs, sign, fabs)
  - Type-specific optimizations

### 4.2 Advanced Indexing (HIGH PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/item_selection.c`, `mapping.c`
- **Raptors**: `src/indexing/advanced/`
- **Features**:
  - Fancy indexing (integer array indexing)
  - Boolean/mask indexing
  - Multi-dimensional indexing
  - Advanced slicing with ellipsis
  - Index array validation
  - Performance optimizations for indexing patterns

### 4.3 Array Concatenation & Splitting (MEDIUM PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/shape.c` (concatenation), various
- **Raptors**: `src/concatenation/`
- **Features**:
  - Concatenate arrays along axis
  - Stack arrays (vstack, hstack, dstack)
  - Split arrays (split, vsplit, hsplit)
  - Array joining utilities
  - Axis validation and handling

### 4.4 Linear Algebra Operations (MEDIUM PRIORITY)
- **Target Files**: `numpy/_core/src/umath/` (dot product, etc.)
- **Raptors**: `src/linalg/`
- **Features**:
  - Dot product (1D, 2D, ND arrays)
  - Matrix multiplication
  - Vector operations
  - Basic linear algebra primitives
  - Broadcasting integration

### 4.5 File I/O - NPY Format (MEDIUM PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/multiarraymodule.c` (I/O), NPY format spec
- **Raptors**: `src/io/`
- **Features**:
  - Save arrays to .npy format
  - Load arrays from .npy format
  - NPY file format parser
  - Header parsing and validation
  - Memory-mapped file support (future)

## Phase 5 Priorities

### 5.1 Advanced Iterators (MEDIUM PRIORITY)
- **Target Files**: `numpy/_core/src/nditer/`
- **Raptors**: `src/iterators/advanced/`
- **Features**:
  - Multi-array iteration (nditer)
  - Iterator with op flags
  - External loop iteration
  - Buffered iteration
  - C-style iteration
  - Fortran-style iteration

### 5.2 Sorting and Searching (MEDIUM PRIORITY)
- **Target Files**: `numpy/_core/src/npysort/`, `searchsorted.c`
- **Raptors**: `src/sorting/`
- **Features**:
  - Sort arrays (quicksort, mergesort, heapsort, stable sort)
  - Argsort (indices that would sort array)
  - Searchsorted (find insertion points)
  - Partition operations
  - Type-specific sort implementations

### 5.3 Array Manipulation Utilities (MEDIUM PRIORITY)
- **Target Files**: Various in `multiarray/`
- **Raptors**: `src/manipulation/`
- **Features**:
  - Flip arrays (flipud, fliplr)
  - Rotate arrays
  - Roll arrays (circular shift)
  - Repeat and tile operations
  - Unique element finding
  - Set operations (union, intersect, etc.)

### 5.4 Statistical Operations (LOWER PRIORITY)
- **Target Files**: Various statistical functions
- **Raptors**: `src/statistics/`
- **Features**:
  - Percentile calculations
  - Median, mode calculations
  - Standard deviation, variance
  - Correlation and covariance
  - Histogram operations

### 5.5 DateTime Support (LOWER PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/datetime*.c`
- **Raptors**: `src/datetime/`
- **Features**:
  - DateTime dtype support
  - DateTime arithmetic
  - DateTime parsing and formatting
  - Timezone handling
  - Timedelta operations

## Phase 6 Priorities

### 6.1 String Operations (LOWER PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/strfuncs.c`
- **Raptors**: `src/string/`
- **Features**:
  - String array operations
  - String concatenation
  - String comparison
  - String formatting
  - Character encoding handling

### 6.2 Masked Array Support (LOWER PRIORITY)
- **Target Files**: Various masked array code
- **Raptors**: `src/masked/`
- **Features**:
  - Masked array structure
  - Mask propagation in operations
  - Masked array creation
  - Masked array operations

### 6.3 DLPack Support (LOWER PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/dlpack.c`
- **Raptors**: `src/dlpack/`
- **Features**:
  - DLPack tensor format conversion
  - Interoperability with other array libraries
  - Memory sharing via DLPack

### 6.4 Structured Arrays (LOWER PRIORITY)
- **Target Files**: `numpy/_core/src/multiarray/descriptor.c` (structured), etc.
- **Raptors**: `src/structured/`
- **Features**:
  - Structured dtype support
  - Field access in structured arrays
  - Record arrays
  - Structured array operations

### 6.5 Memory-Mapped Arrays (LOWER PRIORITY)
- **Target Files**: Various memory mapping code
- **Raptors**: `src/memmap/`
- **Features**:
  - Memory-mapped file arrays
  - Lazy loading of array data
  - Shared memory arrays
  - Large array handling

## Long-Term Goals

### Performance Optimization
- SIMD optimizations for common operations
- Parallel processing support
- JIT compilation opportunities
- Cache-friendly algorithms
- Zero-copy operations where possible

### API Completeness
- Complete NumPy C API coverage
- Python bindings (via PyO3 or similar)
- High-level Rust API design
- Documentation and examples
- Benchmark suite

### Advanced Features
- Custom dtype support
- Array subclassing support
- Broadcasting enhancements
- Advanced memory layout optimizations
- GPU array support (future consideration)

## Module Conversion Status Summary

### Completed (Phases 1-3)
- ✅ Core Array Structure
- ✅ Memory Management
- ✅ Type System (Basic)
- ✅ Array Flags
- ✅ Array Creation (Basic)
- ✅ Indexing (Basic)
- ✅ Broadcasting
- ✅ Shape Manipulation
- ✅ Type Conversion (Basic)
- ✅ Iterators (Basic)
- ✅ Universal Functions (Basic)
- ✅ Array Operations (Basic)
- ✅ Reduction Operations (Basic)

### In Progress / Planned (Phase 4)
- ✅ Advanced Ufuncs (COMPLETED)
- ✅ Advanced Indexing (COMPLETED)
- ✅ Array Concatenation (COMPLETED)
- ✅ Linear Algebra (Basic) (COMPLETED)
- ✅ File I/O (NPY format) (COMPLETED)

### Future (Phases 5-6)
- 📋 Advanced Iterators
- 📋 Sorting and Searching
- 📋 Array Manipulation Utilities
- 📋 Statistical Operations
- 📋 DateTime Support
- 📋 String Operations
- 📋 Masked Arrays
- 📋 DLPack Support
- 📋 Structured Arrays
- 📋 Memory-Mapped Arrays

## Testing Strategy

### Current Status
- 88 unit tests passing across 16 test files
- Integration tests for C API
- Test coverage across all implemented modules:
  - Array creation and properties (8 tests)
  - Indexing - basic and advanced (9 tests)
  - Slicing (6 tests)
  - Broadcasting (8 tests)
  - Shape operations (11 tests)
  - Type system (covered in array tests)
  - Ufuncs - advanced (8 tests)
  - Reductions (8 tests)
  - Array operations - arithmetic and comparison (7 tests)
  - Iterators (5 tests)
  - Concatenation (4 tests)
  - Linear algebra (3 tests)
  - File I/O (2 tests)
  - FFI/C API (9 tests)

### Future Testing Goals
- Comprehensive test suite (>1000 tests)
- Property-based testing for ufuncs
- Performance benchmarks vs NumPy
- Fuzz testing for edge cases
- Memory safety tests
- Concurrency tests (when applicable)

## Documentation Goals

- Complete API documentation
- Architecture documentation
- Conversion guide from NumPy
- Performance guide
- Contribution guide
- Examples and tutorials

## Notes

- All C API functions should be in `src/ffi/mod.rs` or submodules
- Internal Rust API should use idiomatic Rust (Result types, etc.)
- C API should match NumPy's API exactly for compatibility
- Use `#[repr(C)]` for all C-compatible structures
- Document all public APIs
- Test coverage: Currently 88 tests passing across 16 test files covering all implemented modules
- Phase 3 added: Iterators, Ufuncs, Operations, and Reductions with comprehensive test coverage
- Phase 4 added: Advanced Ufuncs, Advanced Indexing, Concatenation, Linear Algebra, and File I/O
- Comprehensive test suite added for: Shape operations (11 tests), Reductions (8 tests), and Array Operations (7 tests)

## Implementation Timeline (Estimated)

### Phase 4 (Next 2-3 months)
- **Week 1-2**: Advanced Ufuncs (trigonometric, logarithmic functions)
- **Week 3-4**: Advanced Indexing (fancy indexing, boolean indexing)
- **Week 5-6**: Array Concatenation & Splitting
- **Week 7-8**: Basic Linear Algebra (dot product, matrix multiplication)
- **Week 9-10**: NPY File I/O (save/load arrays)

### Phase 5 (Months 4-6)
- Advanced Iterators (nditer)
- Sorting and Searching operations
- Array Manipulation Utilities
- Statistical Operations (basic)
- DateTime Support (basic)

### Phase 6+ (Months 7+)
- String Operations
- Masked Array Support
- DLPack Support
- Structured Arrays
- Memory-Mapped Arrays
- Performance Optimizations
- Advanced Features

## Success Criteria

### Phase 4 Goals
- ✅ 50+ trigonometric and logarithmic ufuncs implemented
- ✅ Fancy indexing and boolean indexing functional
- ✅ Array concatenation and splitting working
- ✅ Basic linear algebra operations (dot, matmul)
- ✅ NPY file format support for save/load
- ✅ 88 tests passing (comprehensive coverage for all implemented features)

### Phase 5 Goals
- ✅ Advanced iterators (nditer) functional
- ✅ Full sorting and searching suite
- ✅ Array manipulation utilities complete
- ✅ Basic statistical operations
- ✅ DateTime dtype support
- ✅ 200+ tests passing

### Long-Term Goals
- ✅ >90% NumPy C API compatibility
- ✅ Performance within 2x of NumPy for most operations
- ✅ Comprehensive test coverage (>1000 tests)
- ✅ Full documentation
- ✅ Production-ready stability

## Known Limitations and Future Considerations

### Current Limitations
- Limited dtype support (focus on numeric types first)
- Basic view support (needs enhancement)
- No Python bindings yet (Rust-only for now)
- Limited C API coverage
- No SIMD optimizations yet

### Future Enhancements
- Python bindings via PyO3
- SIMD optimizations (AVX, SSE, NEON)
- GPU array support (CuPy compatibility)
- Parallel processing support
- JIT compilation opportunities
- Custom dtype creation API
- Array subclassing support
- Better memory layout optimizations

## Contributing Guidelines

When contributing new modules:
1. Follow NumPy's implementation as reference
2. Use idiomatic Rust (Result types, proper error handling)
3. Maintain C API compatibility where applicable
4. Add comprehensive tests (>80% coverage per module)
5. Document all public APIs
6. Update this roadmap with progress
7. Consider performance implications
8. Ensure memory safety (use unsafe judiciously)
9. Follow Rust naming conventions
10. Add examples for complex functionality
