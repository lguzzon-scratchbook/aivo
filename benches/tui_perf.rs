use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn bench_fuzzy_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuzzy_search");

    // Generate test data of various sizes
    let sizes = [10, 100, 500, 1000, 5000];

    for size in sizes {
        let items: Vec<String> = (0..size)
            .map(|i| format!("provider-{}-https://api.example-{i}.com/v1", i % 10))
            .collect();

        // Benchmark matches_fuzzy (old &str API, still used by chat_tui)
        group.bench_with_input(
            BenchmarkId::new("matches_fuzzy", size),
            &items,
            |b, items| {
                b.iter(|| {
                    items
                        .iter()
                        .filter(|item| aivo::tui::matches_fuzzy("api", item))
                        .count()
                });
            },
        );

        // Benchmark matches_fuzzy_bytes (HOT PATH - used by FuzzySelect)
        // This is the actual production hot path: query_lower computed once, then byte comparison
        let query_lower: Vec<u8> = "api".bytes().map(|b| b.to_ascii_lowercase()).collect();
        group.bench_with_input(
            BenchmarkId::new("matches_fuzzy_bytes", size),
            &items,
            |b, items| {
                b.iter(|| {
                    items
                        .iter()
                        .filter(|item| {
                            aivo::tui::matches_fuzzy_bytes(&query_lower, item.as_bytes())
                        })
                        .count()
                });
            },
        );

        // Benchmark score_match (&str API)
        group.bench_with_input(BenchmarkId::new("score_match", size), &items, |b, items| {
            b.iter(|| {
                items
                    .iter()
                    .map(|item| aivo::tui::score_match("api", item))
                    .collect::<Vec<_>>()
            });
        });

        // Benchmark score_match_bytes (HOT PATH)
        let query_lower_for_score: Vec<u8> =
            "api".bytes().map(|b| b.to_ascii_lowercase()).collect();
        group.bench_with_input(
            BenchmarkId::new("score_match_bytes", size),
            &items,
            |b, items| {
                b.iter(|| {
                    items
                        .iter()
                        .map(|item| {
                            aivo::tui::score_match_bytes(&query_lower_for_score, item.as_bytes())
                        })
                        .collect::<Vec<_>>()
                });
            },
        );

        // Benchmark full filter + sort workflow (old API)
        group.bench_with_input(
            BenchmarkId::new("filter_and_sort_old", size),
            &items,
            |b, items| {
                b.iter(|| {
                    let mut filtered: Vec<(usize, &String)> = items
                        .iter()
                        .enumerate()
                        .filter(|(_, item)| aivo::tui::matches_fuzzy("api", item))
                        .collect();
                    filtered.sort_by_cached_key(|(_, item)| aivo::tui::score_match("api", item));
                    filtered
                });
            },
        );

        // Benchmark full filter + sort workflow (NEW HOT PATH - as used in FuzzySelect)
        let query_lower_for_filter_sort: Vec<u8> =
            "api".bytes().map(|b| b.to_ascii_lowercase()).collect();
        group.bench_with_input(
            BenchmarkId::new("filter_and_sort", size),
            &items,
            |b, items| {
                b.iter(|| {
                    let mut filtered: Vec<(usize, &String)> = items
                        .iter()
                        .enumerate()
                        .filter(|(_, item)| {
                            aivo::tui::matches_fuzzy_bytes(
                                &query_lower_for_filter_sort,
                                item.as_bytes(),
                            )
                        })
                        .collect();
                    filtered.sort_by_cached_key(|(_, item)| {
                        aivo::tui::score_match_bytes(&query_lower_for_filter_sort, item.as_bytes())
                    });
                    filtered
                });
            },
        );
    }

    // Edge cases: short queries, long queries, case variations
    let items: Vec<String> = (0..1000)
        .map(|i| format!("OpenAI-GPT-4-Provider-{}-https://api.openai.com", i))
        .collect();

    let query_lower_g: Vec<u8> = "g".bytes().map(|b| b.to_ascii_lowercase()).collect();
    group.bench_function("single_char_query_bytes", |b| {
        b.iter(|| {
            items
                .iter()
                .filter(|item| aivo::tui::matches_fuzzy_bytes(&query_lower_g, item.as_bytes()))
                .count()
        });
    });

    let query_lower_long: Vec<u8> = "openai provider"
        .bytes()
        .map(|b| b.to_ascii_lowercase())
        .collect();
    group.bench_function("long_query_bytes", |b| {
        b.iter(|| {
            items
                .iter()
                .filter(|item| aivo::tui::matches_fuzzy_bytes(&query_lower_long, item.as_bytes()))
                .count()
        });
    });

    // Subsequence matching (worst case)
    let query_lower_ogp: Vec<u8> = "ogp".bytes().map(|b| b.to_ascii_lowercase()).collect();
    group.bench_function("subsequence_match_bytes", |b| {
        b.iter(|| {
            items
                .iter()
                .filter(|item| aivo::tui::matches_fuzzy_bytes(&query_lower_ogp, item.as_bytes()))
                .count()
        });
    });

    group.finish();
}

criterion_group! {
    name = tui_benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fuzzy_search
}
criterion_main!(tui_benches);
