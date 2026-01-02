//! Benchmark array operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use raptors_core::{zeros, ones};
use raptors_core::types::{DType, NpyType};
use raptors_core::operations::{add, multiply, divide};
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis};

fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic_operations");
    
    let a = zeros(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    let b = ones(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    group.bench_function("add_100x100", |bencher| {
        bencher.iter(|| {
            black_box(add(black_box(&a), black_box(&b)).unwrap())
        })
    });
    
    group.bench_function("multiply_100x100", |bencher| {
        bencher.iter(|| {
            black_box(multiply(black_box(&a), black_box(&b)).unwrap())
        })
    });
    
    group.bench_function("divide_100x100", |bencher| {
        bencher.iter(|| {
            black_box(divide(black_box(&a), black_box(&b)).unwrap())
        })
    });
    
    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");
    
    let array = ones(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    group.bench_function("sum_axis_0", |bencher| {
        bencher.iter(|| {
            black_box(sum_along_axis(black_box(&array), black_box(Some(0))).unwrap())
        })
    });
    
    group.bench_function("sum_axis_1", |bencher| {
        bencher.iter(|| {
            black_box(sum_along_axis(black_box(&array), black_box(Some(1))).unwrap())
        })
    });
    
    group.bench_function("sum_all", |bencher| {
        bencher.iter(|| {
            black_box(sum_along_axis(black_box(&array), black_box(None)).unwrap())
        })
    });
    
    group.bench_function("mean_axis_0", |bencher| {
        bencher.iter(|| {
            black_box(mean_along_axis(black_box(&array), black_box(Some(0))).unwrap())
        })
    });
    
    group.finish();
}

fn bench_large_arrays(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_array_operations");
    
    let a = zeros(vec![1000, 1000], DType::new(NpyType::Double)).unwrap();
    let b = ones(vec![1000, 1000], DType::new(NpyType::Double)).unwrap();
    
    group.bench_function("add_1000x1000", |bencher| {
        bencher.iter(|| {
            black_box(add(black_box(&a), black_box(&b)).unwrap())
        })
    });
    
    group.bench_function("sum_1000x1000", |bencher| {
        bencher.iter(|| {
            black_box(sum_along_axis(black_box(&a), black_box(None)).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_arithmetic, bench_reductions, bench_large_arrays);
criterion_main!(benches);

