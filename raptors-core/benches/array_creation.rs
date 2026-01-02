//! Benchmark array creation operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use raptors_core::{empty, zeros, ones};
use raptors_core::types::{DType, NpyType};

fn bench_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");
    
    group.bench_function("empty_3x4", |b| {
        b.iter(|| {
            black_box(empty(black_box(vec![3, 4]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("empty_100x100", |b| {
        b.iter(|| {
            black_box(empty(black_box(vec![100, 100]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("zeros_3x4", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![3, 4]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("zeros_100x100", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![100, 100]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("ones_3x4", |b| {
        b.iter(|| {
            black_box(ones(black_box(vec![3, 4]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("ones_100x100", |b| {
        b.iter(|| {
            black_box(ones(black_box(vec![100, 100]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.finish();
}

fn bench_different_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation_dtypes");
    
    group.bench_function("zeros_float32", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![100, 100]), black_box(DType::new(NpyType::Float))).unwrap())
        })
    });
    
    group.bench_function("zeros_float64", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![100, 100]), black_box(DType::new(NpyType::Double))).unwrap())
        })
    });
    
    group.bench_function("zeros_int32", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![100, 100]), black_box(DType::new(NpyType::Int))).unwrap())
        })
    });
    
    group.bench_function("zeros_int64", |b| {
        b.iter(|| {
            black_box(zeros(black_box(vec![100, 100]), black_box(DType::new(NpyType::LongLong))).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_empty, bench_different_dtypes);
criterion_main!(benches);

