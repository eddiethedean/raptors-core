//! Benchmark array indexing operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use raptors_core::zeros;
use raptors_core::types::{DType, NpyType};
use raptors_core::indexing::index_array;

fn bench_integer_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_indexing");
    
    let array = zeros(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    group.bench_function("index_single", |bencher| {
        bencher.iter(|| {
            black_box(index_array(black_box(&array), black_box(&[50, 50])).unwrap())
        })
    });
    
    group.bench_function("index_multiple", |bencher| {
        let indices = vec![vec![10, 20, 30], vec![40, 50, 60]];
        bencher.iter(|| {
            for i in 0..3 {
                for j in 0..3 {
                    black_box(index_array(black_box(&array), black_box(&[indices[0][i], indices[1][j]])).unwrap());
                }
            }
        })
    });
    
    group.finish();
}

// Slice indexing benchmark removed - slicing module is private
// TODO: Make slicing public or add public API for slicing

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration");
    
    let array = zeros(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    group.bench_function("flat_iteration", |bencher| {
        use raptors_core::iterators::FlatIterator;
        bencher.iter(|| {
            let mut iter = FlatIterator::new(black_box(&array));
            let mut count = 0;
            while iter.next() {
                count += 1;
            }
            black_box(count)
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_integer_indexing, bench_iteration);
criterion_main!(benches);

