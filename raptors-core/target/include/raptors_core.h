/* Generated C header for raptors-core */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Maximum number of dimensions for arrays
 */
#define MAXDIMS 64

/**
 * Cache line size in bytes (typically 64 bytes on modern CPUs)
 */
#define CACHE_LINE_SIZE 64

/**
 * L1 cache size (typical value, may vary by CPU)
 */
#define L1_CACHE_SIZE (32 * 1024)

/**
 * L2 cache size (typical value, may vary by CPU)
 */
#define L2_CACHE_SIZE (256 * 1024)

/**
 * Minimum array size threshold for parallelization (default: 10,000 elements)
 * Arrays smaller than this will use sequential operations
 */
#define PARALLEL_THRESHOLD 10000

/**
 * C-compatible array object structure
 *
 * This matches NumPy's PyArrayObject structure for C API compatibility.
 * Fields are public for C API compatibility.
 */
typedef struct PyArrayObject {
  /**
   * Object header (for Python compatibility, will be NULL in pure C usage)
   */
  void *ob_base;
  /**
   * Data pointer
   */
  uint8_t *data;
  /**
   * Number of dimensions
   */
  int nd;
  /**
   * Type descriptor (simplified for now)
   */
  void *descr;
  /**
   * Flags
   */
  uint32_t flags;
  /**
   * Shape array (MAXDIMS elements)
   */
  int64_t dimensions[64];
  /**
   * Strides array (MAXDIMS elements)
   */
  int64_t strides[64];
  /**
   * Base object
   */
  struct PyArrayObject *base;
  /**
   * Descriptor for the array element type
   */
  void *_descr;
  /**
   * Weak references (for Python compatibility)
   */
  void *weakreflist;
} PyArrayObject;

/**
 * Array creation function
 *
 * Creates a new array with the specified parameters.
 *
 * # Safety
 * The caller must ensure `_dimensions` points to an array of at least `_nd` elements if not null.
 * The caller must ensure `_strides` points to an array of at least `_nd` elements if not null.
 */
struct PyArrayObject *PyArray_New(void *_subtype,
                                  int _nd,
                                  const int64_t *_dimensions,
                                  int _type_num,
                                  const int64_t *_strides,
                                  void *_data,
                                  int _itemsize,
                                  int _flags,
                                  void *_obj);

/**
 * Get array size
 *
 * Returns the total number of elements in the array.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
size_t PyArray_SIZE(struct PyArrayObject *arr);

/**
 * Check if object is an array
 *
 * Equivalent to NumPy's PyArray_Check function.
 * Checks if the object is a PyArrayObject (or subclass).
 *
 * # Safety
 * The caller must ensure `op` is a valid pointer if not null.
 */
int PyArray_Check(void *op);

/**
 * Get the number of dimensions of an array
 *
 * Equivalent to NumPy's PyArray_NDIM macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
int PyArray_NDIM(struct PyArrayObject *arr);

/**
 * Get the size of a specific dimension
 *
 * Equivalent to NumPy's PyArray_DIM macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
int64_t PyArray_DIM(struct PyArrayObject *arr, int idim);

/**
 * Get the stride of a specific dimension
 *
 * Equivalent to NumPy's PyArray_STRIDE macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
int64_t PyArray_STRIDE(struct PyArrayObject *arr, int istride);

/**
 * Get the data pointer of an array
 *
 * Equivalent to NumPy's PyArray_DATA macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
void *PyArray_DATA(struct PyArrayObject *arr);

/**
 * Get the item size in bytes
 *
 * Equivalent to NumPy's PyArray_ITEMSIZE macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
size_t PyArray_ITEMSIZE(struct PyArrayObject *arr);

/**
 * Get pointer to dimensions array
 *
 * Equivalent to NumPy's PyArray_DIMS macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
const int64_t *PyArray_DIMS(struct PyArrayObject *arr);

/**
 * Get pointer to strides array
 *
 * Equivalent to NumPy's PyArray_STRIDES macro/function
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject
 */
const int64_t *PyArray_STRIDES(struct PyArrayObject *arr);

/**
 * Create an empty array (C API)
 *
 * Equivalent to NumPy's PyArray_Empty function
 * Creates an array with uninitialized memory
 *
 * # Safety
 * The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
 */
struct PyArrayObject *PyArray_Empty(int _nd, const int64_t *dims, int type_num, int _is_f_order);

/**
 * Create a zero-filled array (C API)
 *
 * Equivalent to NumPy's PyArray_Zeros function
 *
 * # Safety
 * The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
 */
struct PyArrayObject *PyArray_Zeros(int _nd, const int64_t *dims, int type_num, int _is_f_order);

/**
 * Create a one-filled array (C API)
 *
 * Equivalent to NumPy's PyArray_Ones function
 *
 * # Safety
 * The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
 */
struct PyArrayObject *PyArray_Ones(int _nd, const int64_t *dims, int type_num, int _is_f_order);

/**
 * Create array from descriptor
 *
 * Equivalent to NumPy's PyArray_NewFromDescr function.
 *
 * # Safety
 * The caller must ensure `_descr` is a valid descriptor pointer (simplified - not fully implemented).
 * The caller must ensure `_dimensions` points to an array of at least `_nd` elements if not null.
 */
struct PyArrayObject *PyArray_NewFromDescr(void *_subtype,
                                           void *_descr,
                                           int _nd,
                                           const int64_t *_dimensions,
                                           const int64_t *_strides,
                                           void *_data,
                                           int _flags,
                                           void *_obj);

/**
 * Check if object is exactly an array type
 *
 * Equivalent to NumPy's PyArray_CheckExact function
 */
int PyArray_CheckExact(void *op);

/**
 * Create array view with new dtype
 *
 * Equivalent to NumPy's PyArray_View function.
 * Creates a new view of the array with a different dtype.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller is responsible for freeing the returned pointer using appropriate memory management.
 */
struct PyArrayObject *PyArray_View(struct PyArrayObject *arr, void *_descr, void *_type);

/**
 * Create new view with different shape/strides
 *
 * Equivalent to NumPy's PyArray_NewView function.
 * Creates a new view with different shape and strides.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_NewView(struct PyArrayObject *arr, void *_type, void *_descr);

/**
 * Remove dimensions of size 1
 *
 * Equivalent to NumPy's PyArray_Squeeze function.
 * Removes dimensions of size 1 from the array.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_Squeeze(struct PyArrayObject *arr);

/**
 * Flatten array to 1D
 *
 * Equivalent to NumPy's PyArray_Flatten function.
 * Returns a flattened copy of the array.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_Flatten(struct PyArrayObject *arr, int _order);

/**
 * Reshape array to new shape
 *
 * Equivalent to NumPy's PyArray_Reshape function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller must ensure `newshape` points to an array of at least `nd` elements.
 */
struct PyArrayObject *PyArray_Reshape(struct PyArrayObject *arr, const int64_t *newshape, int nd);

/**
 * Transpose array
 *
 * Equivalent to NumPy's PyArray_Transpose function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_Transpose(struct PyArrayObject *arr, const int *_perm);

/**
 * Return flattened view
 *
 * Equivalent to NumPy's PyArray_Ravel function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_Ravel(struct PyArrayObject *arr, int _order);

/**
 * Swap two axes
 *
 * Equivalent to NumPy's PyArray_SwapAxes function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_SwapAxes(struct PyArrayObject *arr, int axis1, int axis2);

/**
 * Take elements using index array
 *
 * Equivalent to NumPy's PyArray_Take function.
 *
 * # Safety
 * The caller must ensure `arr` and `indices` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_Take(struct PyArrayObject *arr,
                                   struct PyArrayObject *indices,
                                   int _axis,
                                   struct PyArrayObject *_out,
                                   int _mode);

/**
 * Put values using index array
 *
 * Equivalent to NumPy's PyArray_Put function.
 *
 * # Safety
 * The caller must ensure `arr`, `indices`, and `values` are valid pointers to PyArrayObject.
 */
int PyArray_Put(struct PyArrayObject *arr,
                struct PyArrayObject *indices,
                struct PyArrayObject *values,
                int _mode);

/**
 * Put values using boolean mask
 *
 * Equivalent to NumPy's PyArray_PutMask function.
 *
 * # Safety
 * The caller must ensure `arr`, `mask`, and `values` are valid pointers to PyArrayObject.
 */
int PyArray_PutMask(struct PyArrayObject *arr,
                    struct PyArrayObject *mask,
                    struct PyArrayObject *values);

/**
 * Choose elements from arrays
 *
 * Equivalent to NumPy's PyArray_Choose function.
 *
 * # Safety
 * The caller must ensure `arr` and `choices` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_Choose(struct PyArrayObject *arr,
                                     struct PyArrayObject *choices,
                                     struct PyArrayObject *_out,
                                     int _mode);

/**
 * Select elements using condition
 *
 * Equivalent to NumPy's PyArray_Compress function.
 *
 * # Safety
 * The caller must ensure `arr` and `condition` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_Compress(struct PyArrayObject *arr,
                                       struct PyArrayObject *condition,
                                       int _axis,
                                       struct PyArrayObject *_out);

/**
 * Concatenate arrays along axis
 *
 * Equivalent to NumPy's PyArray_Concatenate function.
 *
 * # Safety
 * The caller must ensure `arrays` points to an array of at least `n` valid PyArrayObject pointers.
 */
struct PyArrayObject *PyArray_Concatenate(struct PyArrayObject **arrays, int n, int axis);

/**
 * Stack arrays along axis
 *
 * Equivalent to NumPy's PyArray_Stack function.
 *
 * # Safety
 * The caller must ensure `arrays` points to an array of at least `n` valid PyArrayObject pointers.
 */
struct PyArrayObject *PyArray_Stack(struct PyArrayObject **arrays, int n, int axis);

/**
 * Split array into multiple arrays
 *
 * Equivalent to NumPy's PyArray_Split function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller must ensure `indices_or_sections` points to valid data.
 * The caller is responsible for freeing the returned array of pointers.
 */
struct PyArrayObject **PyArray_Split(struct PyArrayObject *arr,
                                     const int64_t *indices_or_sections,
                                     int n,
                                     int axis);

/**
 * Sort array in-place
 *
 * Equivalent to NumPy's PyArray_Sort function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
int PyArray_Sort(struct PyArrayObject *arr, int _axis, int _kind);

/**
 * Return indices that would sort array
 *
 * Equivalent to NumPy's PyArray_ArgSort function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_ArgSort(struct PyArrayObject *arr, int _axis, int _kind);

/**
 * Find insertion points in sorted array
 *
 * Equivalent to NumPy's PyArray_SearchSorted function.
 *
 * # Safety
 * The caller must ensure `arr` and `values` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_SearchSorted(struct PyArrayObject *arr,
                                           struct PyArrayObject *values,
                                           int _side,
                                           struct PyArrayObject *_sorter);

/**
 * Partition array
 *
 * Equivalent to NumPy's PyArray_Partition function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
int PyArray_Partition(struct PyArrayObject *arr, int kth, int _axis, int _kind);

/**
 * Matrix multiplication
 *
 * Equivalent to NumPy's PyArray_MatrixProduct function.
 *
 * # Safety
 * The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_MatrixProduct(struct PyArrayObject *arr1, struct PyArrayObject *arr2);

/**
 * Inner product
 *
 * Equivalent to NumPy's PyArray_InnerProduct function.
 *
 * # Safety
 * The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_InnerProduct(struct PyArrayObject *arr1, struct PyArrayObject *arr2);

/**
 * Matrix multiplication (alias)
 *
 * Equivalent to NumPy's PyArray_MatMul function.
 *
 * # Safety
 * The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_MatMul(struct PyArrayObject *arr1, struct PyArrayObject *arr2);

/**
 * Save array to NPY file
 *
 * Equivalent to NumPy's PyArray_Save function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller must ensure `filename` is a valid null-terminated C string.
 */
int PyArray_Save(struct PyArrayObject *arr, const char *filename, int _format);

/**
 * Load array from NPY file
 *
 * Equivalent to NumPy's PyArray_Load function.
 *
 * # Safety
 * The caller must ensure `filename` is a valid null-terminated C string.
 */
struct PyArrayObject *PyArray_Load(const char *filename);

/**
 * Save array to text file
 *
 * Equivalent to NumPy's PyArray_SaveText function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller must ensure `filename` is a valid null-terminated C string.
 * The caller must ensure `delimiter` is a valid null-terminated C string or null (defaults to space).
 */
int PyArray_SaveText(struct PyArrayObject *arr,
                     const char *filename,
                     const char *delimiter);

/**
 * Load array from text file
 *
 * Equivalent to NumPy's PyArray_LoadText function.
 *
 * # Safety
 * The caller must ensure `filename` is a valid null-terminated C string.
 * The caller must ensure `delimiter` is a valid null-terminated C string or null (auto-detect).
 */
struct PyArrayObject *PyArray_LoadText(const char *filename, const char *delimiter, int skiprows);

/**
 * Broadcast arrays
 *
 * Equivalent to NumPy's PyArray_Broadcast function.
 *
 * # Safety
 * The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_Broadcast(struct PyArrayObject *arr1, struct PyArrayObject *arr2);

/**
 * Broadcast to specific shape
 *
 * Equivalent to NumPy's PyArray_BroadcastToShape function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 * The caller must ensure `shape` points to an array of at least `nd` elements.
 */
struct PyArrayObject *PyArray_BroadcastToShape(struct PyArrayObject *arr,
                                               const int64_t *shape,
                                               int nd);

/**
 * Clip values to range
 *
 * Equivalent to NumPy's PyArray_Clip function.
 *
 * # Safety
 * The caller must ensure `arr`, `min`, and `max` are valid pointers to PyArrayObject.
 */
struct PyArrayObject *PyArray_Clip(struct PyArrayObject *arr,
                                   struct PyArrayObject *min,
                                   struct PyArrayObject *max,
                                   struct PyArrayObject *_out);

/**
 * Round values
 *
 * Equivalent to NumPy's PyArray_Round function.
 *
 * # Safety
 * The caller must ensure `arr` is a valid pointer to a PyArrayObject.
 */
struct PyArrayObject *PyArray_Round(struct PyArrayObject *arr,
                                    int _decimals,
                                    struct PyArrayObject *_out);

/**
 * Compute Einstein summation (C API)
 *
 * Equivalent to NumPy's einsum functionality.
 *
 * # Safety
 * The caller must ensure `subscripts` is a valid null-terminated C string.
 * The caller must ensure `arrays` is a valid array of at least `num_arrays` PyArrayObject pointers.
 * The caller must ensure all array pointers are valid.
 */
struct PyArrayObject *PyArray_Einsum(const char *subscripts,
                                     struct PyArrayObject *const *arrays,
                                     uintptr_t num_arrays);
