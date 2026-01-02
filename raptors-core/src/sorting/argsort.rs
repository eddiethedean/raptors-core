//! Argsort implementation
//!
//! This module provides argsort functionality,
//! returning indices that would sort an array

use crate::array::Array;
use crate::types::{DType, NpyType};

use super::SortingError;

/// Return indices that would sort an array
///
/// # Arguments
/// * `array` - Array to get sort indices for
/// * `kind` - Sort algorithm to use
///
/// # Returns
/// * `Ok(Array)` - Array of indices that would sort the input
/// * `Err(SortingError)` if argsort fails
pub fn argsort(array: &Array, kind: super::SortKind) -> Result<Array, SortingError> {
    let size = array.size();
    let dtype = DType::new(NpyType::Long);
    let mut indices = Array::new(vec![size as i64], dtype)?;
    
    // Initialize indices with 0..size
    unsafe {
        let indices_ptr = indices.data_ptr_mut() as *mut i64;
        for i in 0..size {
            *indices_ptr.add(i) = i as i64;
        }
    }
    
    // Sort indices based on array values
    match array.dtype().type_() {
        NpyType::Double => argsort_double(array, &mut indices, kind),
        NpyType::Float => argsort_float(array, &mut indices, kind),
        NpyType::Int => argsort_int(array, &mut indices, kind),
        NpyType::Long => argsort_long(array, &mut indices, kind),
        _ => Err(SortingError::UnsupportedType),
    }?;
    
    Ok(indices)
}

/// Argsort for double
fn argsort_double(array: &Array, indices: &mut Array, kind: super::SortKind) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr() as *const f64;
        let indices_ptr = indices.data_ptr_mut() as *mut i64;
        
        // Create vector of (value, index) pairs
        let mut pairs: Vec<(f64, i64)> = (0..size)
            .map(|i| (*data_ptr.add(i), *indices_ptr.add(i)))
            .collect();
        
        // Sort pairs by value
        match kind {
            super::SortKind::Quick | super::SortKind::Stable => {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            super::SortKind::Merge => {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            super::SortKind::Heap => {
                // Use heapsort for pairs
                heapsort_pairs_double(&mut pairs);
            }
        }
        
        // Write sorted indices back
        for (i, (_, idx)) in pairs.iter().enumerate() {
            *indices_ptr.add(i) = *idx;
        }
    }
    
    Ok(())
}

/// Argsort for float
fn argsort_float(array: &Array, indices: &mut Array, kind: super::SortKind) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr() as *const f32;
        let indices_ptr = indices.data_ptr_mut() as *mut i64;
        
        let mut pairs: Vec<(f32, i64)> = (0..size)
            .map(|i| (*data_ptr.add(i), *indices_ptr.add(i)))
            .collect();
        
        match kind {
            super::SortKind::Quick | super::SortKind::Stable => {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            super::SortKind::Merge => {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            super::SortKind::Heap => {
                heapsort_pairs_float(&mut pairs);
            }
        }
        
        for (i, (_, idx)) in pairs.iter().enumerate() {
            *indices_ptr.add(i) = *idx;
        }
    }
    
    Ok(())
}

/// Argsort for int
fn argsort_int(array: &Array, indices: &mut Array, kind: super::SortKind) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr() as *const i32;
        let indices_ptr = indices.data_ptr_mut() as *mut i64;
        
        let mut pairs: Vec<(i32, i64)> = (0..size)
            .map(|i| (*data_ptr.add(i), *indices_ptr.add(i)))
            .collect();
        
        match kind {
            super::SortKind::Quick | super::SortKind::Stable => {
                pairs.sort_by_key(|a| a.0);
            }
            super::SortKind::Merge => {
                pairs.sort_by_key(|a| a.0);
            }
            super::SortKind::Heap => {
                heapsort_pairs_int(&mut pairs);
            }
        }
        
        for (i, (_, idx)) in pairs.iter().enumerate() {
            *indices_ptr.add(i) = *idx;
        }
    }
    
    Ok(())
}

/// Argsort for long
fn argsort_long(array: &Array, indices: &mut Array, kind: super::SortKind) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr() as *const i64;
        let indices_ptr = indices.data_ptr_mut() as *mut i64;
        
        let mut pairs: Vec<(i64, i64)> = (0..size)
            .map(|i| (*data_ptr.add(i), *indices_ptr.add(i)))
            .collect();
        
        match kind {
            super::SortKind::Quick | super::SortKind::Stable => {
                pairs.sort_by_key(|a| a.0);
            }
            super::SortKind::Merge => {
                pairs.sort_by_key(|a| a.0);
            }
            super::SortKind::Heap => {
                heapsort_pairs_long(&mut pairs);
            }
        }
        
        for (i, (_, idx)) in pairs.iter().enumerate() {
            *indices_ptr.add(i) = *idx;
        }
    }
    
    Ok(())
}

/// Heapsort for double pairs
fn heapsort_pairs_double(arr: &mut [(f64, i64)]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_pairs_double(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_pairs_double(arr, i, 0);
    }
}

fn heapify_pairs_double(arr: &mut [(f64, i64)], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left].0 > arr[largest].0 {
        largest = left;
    }
    
    if right < n && arr[right].0 > arr[largest].0 {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_pairs_double(arr, n, largest);
    }
}

/// Heapsort for float pairs
fn heapsort_pairs_float(arr: &mut [(f32, i64)]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_pairs_float(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_pairs_float(arr, i, 0);
    }
}

fn heapify_pairs_float(arr: &mut [(f32, i64)], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left].0 > arr[largest].0 {
        largest = left;
    }
    
    if right < n && arr[right].0 > arr[largest].0 {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_pairs_float(arr, n, largest);
    }
}

/// Heapsort for int pairs
fn heapsort_pairs_int(arr: &mut [(i32, i64)]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_pairs_int(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_pairs_int(arr, i, 0);
    }
}

fn heapify_pairs_int(arr: &mut [(i32, i64)], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left].0 > arr[largest].0 {
        largest = left;
    }
    
    if right < n && arr[right].0 > arr[largest].0 {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_pairs_int(arr, n, largest);
    }
}

/// Heapsort for long pairs
fn heapsort_pairs_long(arr: &mut [(i64, i64)]) {
    let n = arr.len();
    
    for i in (0..n / 2).rev() {
        heapify_pairs_long(arr, n, i);
    }
    
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify_pairs_long(arr, i, 0);
    }
}

fn heapify_pairs_long(arr: &mut [(i64, i64)], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    
    if left < n && arr[left].0 > arr[largest].0 {
        largest = left;
    }
    
    if right < n && arr[right].0 > arr[largest].0 {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify_pairs_long(arr, n, largest);
    }
}

