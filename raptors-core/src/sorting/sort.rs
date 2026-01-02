//! Array sorting implementation
//!
//! This module provides sorting functionality for arrays,
//! equivalent to NumPy's sort operations

use crate::array::{Array, ArrayError};
use crate::types::NpyType;

/// Sort algorithm kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    /// Quicksort (unstable, fast)
    Quick,
    /// Mergesort (stable, slower)
    Merge,
    /// Heapsort (unstable, guaranteed O(n log n))
    Heap,
    /// Stable sort (mergesort variant)
    Stable,
}

/// Sorting error
#[derive(Debug, Clone)]
pub enum SortingError {
    /// Array error
    ArrayError(ArrayError),
    /// Unsupported type for sorting
    UnsupportedType,
}

impl std::fmt::Display for SortingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortingError::ArrayError(e) => write!(f, "Array error: {}", e),
            SortingError::UnsupportedType => write!(f, "Unsupported type for sorting"),
        }
    }
}

impl std::error::Error for SortingError {}

impl From<ArrayError> for SortingError {
    fn from(err: ArrayError) -> Self {
        SortingError::ArrayError(err)
    }
}

/// Sort array in-place
///
/// # Arguments
/// * `array` - Array to sort (must be mutable)
/// * `kind` - Sort algorithm to use
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(SortingError)` if sorting fails
pub fn sort(array: &mut Array, kind: SortKind) -> Result<(), SortingError> {
    match array.dtype().type_() {
        NpyType::Double => sort_double(array, kind),
        NpyType::Float => sort_float(array, kind),
        NpyType::Int => sort_int(array, kind),
        NpyType::Long => sort_long(array, kind),
        _ => Err(SortingError::UnsupportedType),
    }
}

/// Sort double array
fn sort_double(array: &mut Array, kind: SortKind) -> Result<(), SortingError> {
    let size = array.size();
    if size == 0 {
        return Ok(());
    }

    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        
        match kind {
            SortKind::Quick | SortKind::Stable => {
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortKind::Merge => {
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortKind::Heap => {
                // Use heapsort implementation
                heapsort_double(slice);
            }
        }
    }

    Ok(())
}

/// Sort float array
fn sort_float(array: &mut Array, kind: SortKind) -> Result<(), SortingError> {
    let size = array.size();
    if size == 0 {
        return Ok(());
    }

    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        
        match kind {
            SortKind::Quick | SortKind::Stable => {
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortKind::Merge => {
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortKind::Heap => {
                heapsort_float(slice);
            }
        }
    }

    Ok(())
}

/// Sort int array
fn sort_int(array: &mut Array, kind: SortKind) -> Result<(), SortingError> {
    let size = array.size();
    if size == 0 {
        return Ok(());
    }

    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut i32;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        
        match kind {
            SortKind::Quick | SortKind::Stable => {
                slice.sort();
            }
            SortKind::Merge => {
                slice.sort();
            }
            SortKind::Heap => {
                heapsort_int(slice);
            }
        }
    }

    Ok(())
}

/// Sort long array
fn sort_long(array: &mut Array, kind: SortKind) -> Result<(), SortingError> {
    let size = array.size();
    if size == 0 {
        return Ok(());
    }

    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut i64;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        
        match kind {
            SortKind::Quick | SortKind::Stable => {
                slice.sort();
            }
            SortKind::Merge => {
                slice.sort();
            }
            SortKind::Heap => {
                heapsort_long(slice);
            }
        }
    }

    Ok(())
}

/// Heapsort for double
fn heapsort_double(arr: &mut [f64]) {
    let n = arr.len();
    
    // Build heap
    for i in (0..n / 2).rev() {
        heapify_double(arr, n, i);
    }
    
    // Extract elements from heap
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_double(arr, i, 0);
    }
}

fn heapify_double(arr: &mut [f64], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_double(arr, n, largest);
    }
}

/// Heapsort for float
fn heapsort_float(arr: &mut [f32]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_float(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_float(arr, i, 0);
    }
}

fn heapify_float(arr: &mut [f32], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_float(arr, n, largest);
    }
}

/// Heapsort for int
fn heapsort_int(arr: &mut [i32]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_int(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_int(arr, i, 0);
    }
}

fn heapify_int(arr: &mut [i32], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_int(arr, n, largest);
    }
}

/// Heapsort for long
fn heapsort_long(arr: &mut [i64]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_long(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_long(arr, i, 0);
    }
}

fn heapify_long(arr: &mut [i64], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_long(arr, n, largest);
    }
}

