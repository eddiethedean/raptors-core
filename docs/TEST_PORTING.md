# NumPy Test Suite Porting Guide

This document tracks the porting of NumPy's test suite to Raptors.

## NumPy Test Structure Analysis

### Core Test Categories

#### Array Creation and Properties
- **NumPy Files**: `test_creation.py`, `test_array.py`, `test_multiarray.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/creation_test.rs`
- **Estimated Tests**: 50-100
- **Status**: Not Started
- **Dependencies**: Basic array creation, dtype system

#### Array Indexing
- **NumPy Files**: `test_indexing.py`, `test_multiarray.py` (indexing sections)
- **Raptors Target**: `raptors-core/tests/numpy_port/indexing_test.rs`
- **Estimated Tests**: 100-150
- **Status**: Not Started
- **Dependencies**: Basic indexing, advanced indexing, boolean indexing

#### Broadcasting
- **NumPy Files**: `test_broadcasting.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/broadcasting_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: Broadcasting logic, stride calculation

#### Universal Functions (Ufuncs)
- **NumPy Files**: `test_umath.py`, `test_ufunc.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/ufunc_test.rs`
- **Estimated Tests**: 200-300
- **Status**: Not Started
- **Dependencies**: Ufunc system, type promotion, broadcasting

#### Reductions
- **NumPy Files**: `test_reduction.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/reduction_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: Reduction operations, axis handling

#### Shape Manipulation
- **NumPy Files**: `test_shape_base.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/shape_test.rs`
- **Estimated Tests**: 75-100
- **Status**: Not Started
- **Dependencies**: Reshape, transpose, squeeze, expand_dims

#### Array Operations
- **NumPy Files**: `test_array_operations.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/operations_test.rs`
- **Estimated Tests**: 100-150
- **Status**: Not Started
- **Dependencies**: Concatenate, stack, split

#### Linear Algebra
- **NumPy Files**: `test_linalg.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/linalg_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: Dot product, matrix multiplication

#### Type System
- **NumPy Files**: `test_dtype.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/dtype_test.rs`
- **Estimated Tests**: 75-100
- **Status**: Not Started
- **Dependencies**: Dtype creation, promotion, casting

#### Masked Arrays
- **NumPy Files**: `test_ma.py` (if applicable)
- **Raptors Target**: `raptors-core/tests/numpy_port/masked_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: Masked array implementation

#### Structured Arrays
- **NumPy Files**: `test_structured.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/structured_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: Structured dtype, field access

#### String Arrays
- **NumPy Files**: `test_strings.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/string_test.rs`
- **Estimated Tests**: 30-50
- **Status**: Not Started
- **Dependencies**: String dtype, string operations

#### DateTime
- **NumPy Files**: `test_datetime.py`
- **Raptors Target**: `raptors-core/tests/numpy_port/datetime_test.rs`
- **Estimated Tests**: 50-75
- **Status**: Not Started
- **Dependencies**: DateTime dtype, timedelta

### Python API Tests

#### Python Array API
- **NumPy Files**: Python API tests
- **Raptors Target**: `raptors-python/tests/numpy_port/test_array_api.py`
- **Estimated Tests**: 200-300
- **Status**: Not Started

#### Python Ufunc API
- **Raptors Target**: `raptors-python/tests/numpy_port/test_ufunc_api.py`
- **Estimated Tests**: 100-200
- **Status**: Not Started

#### Python Interoperability
- **Raptors Target**: `raptors-python/tests/numpy_port/test_interop.py`
- **Estimated Tests**: 50-100
- **Status**: Not Started

## Test Porting Progress

### Phase 1: Infrastructure ✅
- [x] Analysis document created
- [x] Test porting infrastructure created
- [x] NumPy testing utilities implemented
- [x] Test porting script created

### Phase 2: Core Tests
- [x] Array Creation Tests (65/50-100) ✅
- [x] Array Indexing Tests (42/100-150) ✅
- [x] Broadcasting Tests (35/50-75) ✅
- [x] Ufunc Tests (30/200-300) ✅
- [x] Reduction Tests (30/50-75) ✅
- [x] Shape Manipulation Tests (46/75-100) ✅

### Phase 3: Advanced Tests
- [ ] Shape Manipulation Tests (0/75-100)
- [ ] Array Operations Tests (0/100-150)
- [ ] Linear Algebra Tests (0/50-75)
- [ ] Type System Tests (0/75-100)

### Phase 4: Specialized Tests
- [ ] Masked Array Tests (0/50-75)
- [ ] Structured Array Tests (0/50-75)
- [ ] String Array Tests (0/30-50)
- [ ] DateTime Tests (0/50-75)

### Phase 5: Python API Tests
- [x] Python Array API Tests (60+ tests created) ✅
- [x] Python Ufunc API Tests (30+ tests created) ✅
- [x] Python Interoperability Tests (20+ tests created) ✅

## Test Mapping Table

| NumPy Test File | Raptors Test File | Status | Tests Ported | Notes |
|-----------------|-------------------|--------|--------------|-------|
| test_creation.py | numpy_port_creation_test.rs | Complete | 65/50-100 | Comprehensive creation tests |
| test_indexing.py | numpy_port_indexing_test.rs | In Progress | 42/100-150 | Basic and advanced indexing tests |
| test_broadcasting.py | numpy_port_broadcasting_test.rs | In Progress | 35/50-75 | Comprehensive broadcasting tests |
| test_umath.py | numpy_port_ufunc_test.rs | In Progress | 30/200-300 | Basic ufunc operations |
| test_reduction.py | numpy_port_reduction_test.rs | Complete | 30/50-75 | Comprehensive reduction tests |
| test_shape_base.py | numpy_port_shape_test.rs | Complete | 46/75-100 | Comprehensive shape manipulation tests |
| test_array_operations.py | operations_test.rs | Not Started | 0/100-150 | |
| test_linalg.py | linalg_test.rs | Not Started | 0/50-75 | |
| test_dtype.py | dtype_test.rs | Not Started | 0/75-100 | |

## NumPy Testing Utilities Needed

### numpy.testing Functions to Implement

1. **assert_array_equal** - Element-wise array equality
2. **assert_array_almost_equal** - Floating point comparison with tolerance
3. **assert_array_less** - Element-wise less-than comparison
4. **assert_allclose** - All-close comparison for floating point arrays
5. **assert_almost_equal** - Scalar floating point comparison
6. **assert_raises** - Exception testing helper
7. **assert_warns** - Warning testing helper

### Test Data Generators

1. Random array generators
2. Edge case generators (NaN, Infinity, empty arrays)
3. Shape variation generators
4. Dtype variation generators

## Porting Process

1. **Select test file** from NumPy
2. **Analyze dependencies** (numpy.testing utilities, NumPy features)
3. **Create helper functions** if needed
4. **Port test logic** to Rust/Python
5. **Adapt API calls** (NumPy → Raptors)
6. **Run and fix** until passing
7. **Document** any behavioral differences
8. **Update progress** in this document

## Notes

- Tests should maintain NumPy's behavior patterns
- Document any intentional differences from NumPy
- Prioritize core functionality tests first
- Use hybrid approach: Rust for core, Python for API

## CI/CD Integration

The ported tests are integrated into the CI/CD pipeline (`.github/workflows/ci.yml`):
- Rust tests run on Ubuntu and macOS
- Python tests run on multiple Python versions (3.9, 3.10, 3.11)
- NumPy port tests are run separately for better visibility
- Test categorization and counting is automated

## Test Statistics

### Rust Tests (Ported from NumPy)
- **Creation Tests**: 65 tests
- **Indexing Tests**: 42 tests
- **Broadcasting Tests**: 35 tests
- **Ufunc Tests**: 30 tests
- **Reduction Tests**: 30 tests
- **Shape Tests**: 46 tests
- **Helper Tests**: 7 tests
- **Total Rust Ported**: 255+ tests

### Python Tests (Ported from NumPy)
- **Array API Tests**: 60+ tests
- **Ufunc API Tests**: 30+ tests
- **Interoperability Tests**: 20+ tests
- **Total Python Ported**: 110+ tests

### Overall Progress
- **Total Ported**: 365+ tests (255 Rust + 110 Python)
- **Infrastructure**: Complete
- **Core Tests**: In Progress (6/6 categories started)
- **Python API Tests**: Complete
- **CI/CD Integration**: Complete

## Missing Features and Skipped Tests Registry

This section tracks all missing features identified during test porting and the tests that are skipped until these features are implemented.

### Feature Tracking Table

| Feature | Phase | Location | Status | Tests to Unskip |
|---------|-------|----------|--------|-----------------|
| Python operator overloading (`+`, `-`, `*`, `/`) | 13.1 | `raptors-python/src/array.rs` | ⏳ Planned | 3 tests |
| Array methods (`reshape()`, `flatten()`, `sum()`, etc.) | 13.2 | `raptors-python/src/array.rs` | ⏳ Planned | 6 tests |
| Negative indexing | 13.3 | `raptors-core/src/indexing/indexing.rs` | ⏳ Planned | 1 test |
| Ufunc Python API (`raptors.add()`, etc.) | 13.4 | `raptors-python/src/ufunc.rs` | ⏳ Planned | 7 tests |
| Array Protocol (`__array__`) | 13.5 | `raptors-python/src/array.rs` | ⏳ Planned | 4 tests |
| NumPy interoperability (`from_numpy()`, `to_numpy()`) | 13.6 | `raptors-python/src/numpy_interop.rs` | ⏳ Planned | 4 tests |
| Reduction axis handling | 14.1 | `raptors-core/src/ufunc/reduction.rs` | ⏳ Planned | 1 test update |
| Type promotion in ufuncs | 14.2 | `raptors-core/src/conversion/promotion.rs` | ⏳ Planned | 2 tests |
| NaN/Infinity handling | 14.3 | `raptors-core/src/ufunc/loops.rs` | ⏳ Planned | 2 tests |
| Broadcasting in Python API | 14.4 | `raptors-python/src/array.rs`, `ufunc.rs` | ⏳ Planned | 4 tests |
| Structured array field access | 14.5 | `raptors-core/src/structured/fields.rs` | ⏳ Planned | Tests needed |
| DLPack protocol Python wrapper | 14.6 | `raptors-python/src/array.rs` | ⏳ Planned | 1 test |
| Array slicing (full support) | 13.3 | `raptors-core/src/indexing/slicing.rs` | ⏳ Planned | 1 test |

### Skipped Test Registry

#### Python Array API Tests (`test_array_api.py`)

| Test Name | Skip Reason | Required Feature | Phase | Unskip Criteria |
|-----------|-------------|------------------|-------|-----------------|
| `TestArrayIndexing::test_slicing_1d` | Slicing not yet implemented | Full array slicing support | 13.3 | Implement complete slicing in `slicing.rs` |
| `TestArrayIndexing::test_negative_indexing` | Negative indexing not yet implemented | Negative index normalization | 13.3 | Add negative index support to `indexing.rs` |
| `TestArrayOperations::test_add_arrays` | Array addition not yet implemented | Operator overloading `__add__` | 13.1 | Implement `__add__` in `array.rs` |
| `TestArrayOperations::test_multiply_arrays` | Array multiplication not yet implemented | Operator overloading `__mul__` | 13.1 | Implement `__mul__` in `array.rs` |
| `TestArrayOperations::test_scalar_operations` | Scalar operations not yet implemented | Scalar broadcasting in operators | 13.1 | Add scalar support to operators |
| `TestArrayMethods::test_reshape` | Reshape not yet implemented | `reshape()` method wrapper | 13.2 | Add Python wrapper for Rust reshape |
| `TestArrayMethods::test_flatten` | Flatten not yet implemented | `flatten()` method wrapper | 13.2 | Add Python wrapper for Rust flatten |
| `TestArrayMethods::test_sum` | Sum not yet implemented | `sum()` method wrapper | 13.2 | Add Python wrapper for reduction sum |
| `TestArrayMethods::test_max_min` | Max/min not yet implemented | `max()`, `min()` method wrappers | 13.2 | Add Python wrappers for reductions |
| `TestArrayBroadcasting::test_broadcast_scalar` | Broadcasting not yet implemented | Broadcasting in Python operations | 14.4 | Integrate broadcasting into Python API |
| `TestArrayBroadcasting::test_broadcast_shapes` | Broadcasting not yet implemented | Broadcasting in Python operations | 14.4 | Integrate broadcasting into Python API |
| `TestArrayConversion::test_to_list` | tolist not yet implemented | `tolist()` method | 13.2 | Implement `tolist()` in `array.rs` |
| `TestArrayConversion::test_to_numpy` | to_numpy not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestArrayDtypeOperations::test_dtype_conversion` | astype not yet implemented | `astype()` method wrapper | 13.2 | Add Python wrapper for type casting |

#### Python Ufunc API Tests (`test_ufunc_api.py`)

| Test Name | Skip Reason | Required Feature | Phase | Unskip Criteria |
|-----------|-------------|------------------|-------|-----------------|
| `TestUfuncBasic::test_add_ufunc` | add ufunc not yet implemented | `raptors.add()` function | 13.4 | Implement `add()` in `ufunc.rs` |
| `TestUfuncBasic::test_subtract_ufunc` | subtract ufunc not yet implemented | `raptors.subtract()` function | 13.4 | Implement `subtract()` in `ufunc.rs` |
| `TestUfuncBasic::test_multiply_ufunc` | multiply ufunc not yet implemented | `raptors.multiply()` function | 13.4 | Implement `multiply()` in `ufunc.rs` |
| `TestUfuncBasic::test_divide_ufunc` | divide ufunc not yet implemented | `raptors.divide()` function | 13.4 | Implement `divide()` in `ufunc.rs` |
| `TestUfuncBroadcasting::test_add_broadcast_scalar` | Broadcasting in ufuncs not yet implemented | Broadcasting in ufunc Python API | 14.4 | Integrate broadcasting into ufunc calls |
| `TestUfuncBroadcasting::test_multiply_broadcast_shapes` | Broadcasting in ufuncs not yet implemented | Broadcasting in ufunc Python API | 14.4 | Integrate broadcasting into ufunc calls |
| `TestUfuncTypePromotion::test_int_float_promotion` | Type promotion not yet implemented | Type promotion in ufuncs | 14.2 | Verify/enhance promotion logic |
| `TestUfuncTypePromotion::test_same_type_preservation` | Type promotion not yet implemented | Type promotion in ufuncs | 14.2 | Verify/enhance promotion logic |
| `TestUfuncEdgeCases::test_empty_array_ufunc` | Empty array ufuncs not yet implemented | Empty array handling | - | Verify empty array support |
| `TestUfuncEdgeCases::test_zero_dimensional_ufunc` | 0-d array ufuncs not yet implemented | 0-d array handling | - | Verify 0-d array support |
| `TestUfuncEdgeCases::test_nan_handling` | NaN handling not yet implemented | NaN handling in ufunc loops | 14.3 | Add NaN handling to loops |
| `TestUfuncEdgeCases::test_infinity_handling` | Infinity handling not yet implemented | Infinity handling in ufunc loops | 14.3 | Add Infinity handling to loops |
| `TestUfuncComparison::test_equal_ufunc` | equal ufunc not yet implemented | `raptors.equal()` function | 13.4 | Implement `equal()` in `ufunc.rs` |
| `TestUfuncComparison::test_less_ufunc` | less ufunc not yet implemented | `raptors.less()` function | 13.4 | Implement `less()` in `ufunc.rs` |
| `TestUfuncComparison::test_greater_ufunc` | greater ufunc not yet implemented | `raptors.greater()` function | 13.4 | Implement `greater()` in `ufunc.rs` |

#### Python Interoperability Tests (`test_interop.py`)

| Test Name | Skip Reason | Required Feature | Phase | Unskip Criteria |
|-----------|-------------|------------------|-------|-----------------|
| `TestArrayConversion::test_numpy_to_raptors` | NumPy to Raptors conversion not yet implemented | `from_numpy()` function | 13.6 | Implement `from_numpy()` in `numpy_interop.rs` |
| `TestArrayConversion::test_raptors_to_numpy` | Raptors to NumPy conversion not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestArrayConversion::test_roundtrip_conversion` | Roundtrip conversion not yet implemented | Both `from_numpy()` and `__array__` | 13.5, 13.6 | Implement both features |
| `TestSharedMemory::test_memory_sharing` | Memory sharing not yet implemented | Memory sharing detection | - | Add memory sharing tracking |
| `TestNumPyFunctionsWithRaptors::test_numpy_sum` | NumPy functions with Raptors arrays not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestNumPyFunctionsWithRaptors::test_numpy_mean` | NumPy functions with Raptors arrays not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestNumPyFunctionsWithRaptors::test_numpy_max_min` | NumPy functions with Raptors arrays not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestProtocolCompliance::test_array_protocol` | Array protocol not yet implemented | Array Protocol `__array__` | 13.5 | Implement `__array__` method |
| `TestProtocolCompliance::test_dlpack_protocol` | DLPack protocol not yet implemented | DLPack protocol `__dlpack__` | 14.6 | Add Python wrapper for DLPack |
| `TestTypeCompatibility::test_dtype_compatibility` | Dtype compatibility not yet implemented | NumPy dtype conversion | 13.6 | Implement dtype conversion |
| `TestTypeCompatibility::test_shape_compatibility` | Shape compatibility not yet implemented | Shape conversion | 13.6 | Verify shape handling |

#### Rust Core Tests

| Test Name | File | Issue | Required Feature | Phase | Fix Criteria |
|-----------|------|-------|------------------|-------|--------------|
| `test_sum_along_axis_axis_0` | `reduction_test.rs` | TODO comment - axis handling incomplete | Proper axis-specific reduction | 14.1 | Fix axis reduction logic in `reduction.rs` |

### Feature Implementation Priority

**High Priority (Phase 13 - Python API Completeness)**
1. Array operator overloading (enables basic Python usage)
2. Array methods (enables NumPy-like API)
3. Array Protocol (enables NumPy interoperability)
4. NumPy interoperability functions (enables seamless integration)

**Medium Priority (Phase 14 - Core Enhancements)**
1. Reduction axis handling (fixes existing TODO)
2. Type promotion verification (ensures correctness)
3. Broadcasting integration (enables advanced operations)

**Lower Priority**
1. Special values handling (NaN/Infinity) - important but less critical
2. Structured array field access - specialized feature
3. DLPack protocol - advanced interoperability

### Unskip Process

When implementing a feature:
1. Implement the feature in the specified location
2. Add tests for the feature
3. Find all skipped tests in this registry that depend on the feature
4. Remove `pytest.skip()` or `skip` annotations from those tests
5. Run the tests to verify they pass
6. Update this registry to mark tests as unskipped
7. Update the feature status in the Feature Tracking Table

