use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn bench_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto");

    // Test different payload sizes
    let sizes = [32, 128, 512, 2048, 8192];

    for size in sizes {
        let plaintext = "A".repeat(size);

        group.bench_with_input(BenchmarkId::new("encrypt", size), &plaintext, |b, text| {
            b.iter(|| aivo::services::session_crypto::encrypt(text));
        });

        // Pre-encrypt for decryption benchmarks
        let encrypted = aivo::services::session_crypto::encrypt(&plaintext).unwrap();

        group.bench_with_input(BenchmarkId::new("decrypt", size), &encrypted, |b, enc| {
            b.iter(|| aivo::services::session_crypto::decrypt(enc));
        });

        // Roundtrip benchmark
        group.bench_with_input(
            BenchmarkId::new("roundtrip", size),
            &plaintext,
            |b, text| {
                b.iter(|| {
                    let enc = aivo::services::session_crypto::encrypt(text).unwrap();
                    aivo::services::session_crypto::decrypt(&enc).unwrap()
                });
            },
        );
    }

    // Idempotency check - already encrypted data
    let encrypted = aivo::services::session_crypto::encrypt("test-data-123").unwrap();
    group.bench_function("encrypt_idempotent", |b| {
        b.iter(|| aivo::services::session_crypto::encrypt(&encrypted));
    });

    group.finish();
}

criterion_group! {
    name = crypto_benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_crypto
}
criterion_main!(crypto_benches);
