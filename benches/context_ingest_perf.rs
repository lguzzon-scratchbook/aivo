use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};
use serde_json::json;

fn bench_context_sanitization(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_ingest");

    // Benchmark sanitize_turn with various text sizes
    let sizes = [100, 500, 1000, 4000, 10000];

    for size in sizes {
        let text = format!(
            "Please review the pagination approach in handlers/users.go {} # aivo memory\n## [claude] old context",
            "x".repeat(size)
        );

        group.bench_with_input(BenchmarkId::new("sanitize_turn", size), &text, |b, t| {
            b.iter(|| aivo::services::context_ingest::sanitize_turn(t))
        });
    }

    // Benchmark is_substantive
    let substantive_text = "Please review the pagination approach in handlers/users.go and ensure the cursor handling is correct for empty results.";
    let short_text = "ok";
    let boilerplate_text = "<local-command-caveat> Some short caveat here";

    group.bench_function("is_substantive_true", |b| {
        b.iter(|| aivo::services::context_ingest::is_substantive(substantive_text));
    });

    group.bench_function("is_substantive_short", |b| {
        b.iter(|| aivo::services::context_ingest::is_substantive(short_text));
    });

    group.bench_function("is_substantive_boilerplate", |b| {
        b.iter(|| aivo::services::context_ingest::is_substantive(boilerplate_text));
    });

    // Benchmark extract_claude_text with different message formats
    let simple_message = json!({"content": "Hello world, this is a test message"});
    let array_message = json!({
        "content": [
            {"type": "text", "text": "First part of the message"},
            {"type": "tool_use", "id": "tool-1"},
            {"type": "text", "text": "Second part after tool use"}
        ]
    });
    let complex_message = json!({
        "content": [
            {"type": "text", "text": "Text block 1"},
            {"type": "tool_use", "id": "tool-1"},
            {"type": "text", "text": "Text block 2"},
            {"type": "tool_result", "content": "result"},
            {"type": "text", "text": "Text block 3"},
            {"type": "thinking", "thinking": "internal thought"},
            {"type": "text", "text": "Final text"}
        ]
    });

    group.bench_function("extract_claude_text_simple", |b| {
        b.iter(|| aivo::services::context_ingest::extract_claude_text(Some(&simple_message)));
    });

    group.bench_function("extract_claude_text_array", |b| {
        b.iter(|| aivo::services::context_ingest::extract_claude_text(Some(&array_message)));
    });

    group.bench_function("extract_claude_text_complex", |b| {
        b.iter(|| aivo::services::context_ingest::extract_claude_text(Some(&complex_message)));
    });

    // Benchmark extract_codex_message_text
    let codex_payload = json!({
        "content": [
            {"type": "input_text", "text": "User input here"},
            {"type": "output_text", "text": "Assistant output"},
            {"type": "text", "text": "Generic text block"},
            {"type": "tool_call", "id": "call-1"}
        ]
    });

    group.bench_function("extract_codex_text", |b| {
        b.iter(|| aivo::services::context_ingest::extract_codex_message_text(&codex_payload));
    });

    // Benchmark encode_claude_dir
    let paths = [
        "/Users/yc/project/aivo",
        "/home/user/projects/my-application/src/components",
        "C:\\Users\\alice\\repo",
        "\\\\?\\C:\\Users\\alice\\repo",
    ];

    for (i, path) in paths.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("encode_claude_dir", i), *path, |b, p| {
            b.iter(|| aivo::services::context_ingest::encode_claude_dir(p))
        });
    }

    // Benchmark strip_aivo_context with markers at various positions
    let with_early_marker = "Hello world. # aivo memory\nOld context here that should be stripped";
    let with_late_marker = format!("{}# aivo memory\nOld context", "x".repeat(5000));
    let without_marker = "Just plain text without any markers at all in the content";

    group.bench_function("strip_context_early", |b| {
        b.iter(|| aivo::services::context_ingest::strip_aivo_context(with_early_marker));
    });

    group.bench_function("strip_context_late", |b| {
        b.iter(|| aivo::services::context_ingest::strip_aivo_context(&with_late_marker));
    });

    group.bench_function("strip_context_none", |b| {
        b.iter(|| aivo::services::context_ingest::strip_aivo_context(without_marker));
    });

    group.finish();
}

criterion_group! {
    name = context_benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_context_sanitization
}
criterion_main!(context_benches);
