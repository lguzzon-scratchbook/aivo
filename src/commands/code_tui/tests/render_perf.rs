//! Render-path timing probes, not assertions: run with
//! `cargo test --release --features __internal_test_fast_crypto render_perf -- --ignored --nocapture`
//! and compare the printed per-frame costs across changes.

use super::super::*;
use super::helpers::make_test_app;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use std::time::Instant;

/// A realistic assistant reply: prose, a fenced code block, and a list.
fn reply_block(i: usize) -> String {
    format!(
        "Looking at request #{i}, the router resolves the provider in three steps \
and falls back to the models cache when the probe times out.\n\n\
```rust\nfn resolve_{i}(key: &ApiKey) -> Route {{\n    let probe = probe_provider(key);\n    \
match probe {{\n        Ok(route) => route,\n        Err(_) => cached_route(key),\n    }}\n}}\n```\n\n\
- probe the native endpoint first\n- then the OpenAI-compatible bridge\n- finally the cached route from the last run\n\n\
The important part is that step {i} never blocks the event loop.",
    )
}

fn seed_large_history(app: &mut CodeTuiApp, exchanges: usize) {
    for i in 0..exchanges {
        app.history.push(ChatMessage {
            model: None,
            role: "user".to_string(),
            content: format!("question {i}: how does the provider router pick a route?"),
            reasoning_content: None,
            attachments: vec![],
        });
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: reply_block(i),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

fn time_frames(
    label: &str,
    app: &mut CodeTuiApp,
    terminal: &mut Terminal<TestBackend>,
    frames: u32,
    mut per_frame: impl FnMut(&mut CodeTuiApp),
) {
    // Warm the caches so the loop measures steady-state frames.
    terminal.draw(|frame| app.render(frame)).unwrap();
    let start = Instant::now();
    for _ in 0..frames {
        per_frame(app);
        app.frame_tick = app.frame_tick.wrapping_add(1);
        terminal.draw(|frame| app.render(frame)).unwrap();
    }
    let total = start.elapsed();
    println!(
        "{label}: {frames} frames in {total:?} → {:?}/frame",
        total / frames
    );
}

/// Steady-state frame with a large committed history and nothing animating —
/// the cost a keystroke repaint pays in a long session.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_idle_frame_large_history() {
    for exchanges in [25usize, 100, 400] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        seed_large_history(&mut app, exchanges);
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let label = format!("idle {exchanges}-exchange history");
        time_frames(&label, &mut app, &mut terminal, 120, |_| {});
    }
}

/// Spinner-animation frame mid-turn: large history plus a sizeable streamed
/// reply, no new content — the ~60fps steady state while a turn runs.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_animating_frame_mid_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 100);
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.pending_response = (0..40).map(reply_block).collect::<Vec<_>>().join("\n\n");
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    time_frames("animating mid-turn", &mut app, &mut terminal, 120, |_| {});
}

/// Where a typewriter tick's cost goes: markdown render vs styled wrap vs
/// plain prepass of the volatile tail, at a large reply size.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_tail_stage_split() {
    use super::super::render::{wrap_plain_lines, wrap_transcript};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 1);
    app.sending = true;
    app.pending_response = (0..150).map(reply_block).collect::<Vec<_>>().join("\n\n");
    println!("reply bytes: {}", app.pending_response.len());
    let width = 116u16;

    let start = Instant::now();
    let mut blocks = (Vec::new(), Vec::new());
    for _ in 0..30 {
        blocks = app.volatile_tail_blocks(width);
    }
    println!(
        "volatile_tail_blocks (markdown): {:?}/call",
        start.elapsed() / 30
    );

    let (lines, bars) = blocks;
    let start = Instant::now();
    for _ in 0..30 {
        std::hint::black_box(wrap_transcript(&lines, &bars, width));
    }
    println!(
        "wrap_transcript (styled wrap): {:?}/call",
        start.elapsed() / 30
    );

    let plain: Vec<String> = lines.iter().map(|l| l.plain.clone()).collect();
    let start = Instant::now();
    for _ in 0..30 {
        std::hint::black_box(wrap_plain_lines(&plain, width).len());
    }
    println!(
        "wrap_plain_lines (prepass): {:?}/call",
        start.elapsed() / 30
    );
}

/// Scaling curve of the tail markdown render + wrap across reply sizes.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_tail_scaling_curve() {
    use super::super::render::wrap_transcript;
    let width = 116u16;
    for blocks in [25usize, 50, 100, 150, 200] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        seed_large_history(&mut app, 1);
        app.sending = true;
        app.pending_response = (0..blocks)
            .map(reply_block)
            .collect::<Vec<_>>()
            .join("\n\n");
        let bytes = app.pending_response.len();
        let start = Instant::now();
        let (lines, bars) = app.volatile_tail_blocks(width);
        let md = start.elapsed();
        let start = Instant::now();
        std::hint::black_box(wrap_transcript(&lines, &bars, width));
        let wrap = start.elapsed();
        println!(
            "{blocks} blocks ({bytes} B, {} lines): md {md:?}, wrap {wrap:?}",
            lines.len()
        );
    }
}

#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_real_session() {
    use super::super::render::wrap_transcript;
    let Ok(path) = std::env::var("AIVO_PERF_SESSION") else {
        println!("skip bench_real_session: set AIVO_PERF_SESSION to a session.json path");
        return;
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        println!("skip bench_real_session: {path} not found");
        return;
    };
    let session: serde_json::Value = serde_json::from_str(&raw).expect("session json");
    let messages = session["messages"].as_array().expect("messages");
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let field =
        |m: &serde_json::Value, k: &str| m.get(k).and_then(|v| v.as_str()).map(str::to_string);
    for m in messages {
        app.history.push(ChatMessage {
            model: field(m, "model"),
            role: field(m, "role").unwrap_or_else(|| "assistant".to_string()),
            content: field(m, "content").unwrap_or_default(),
            reasoning_content: field(m, "reasoning_content"),
            attachments: vec![],
        });
    }
    println!(
        "session {path}: {} messages, {} content bytes",
        app.history.len(),
        app.history.iter().map(|m| m.content.len()).sum::<usize>()
    );

    let width = 116u16;
    let start = Instant::now();
    let body = app.build_transcript_history_body(width);
    println!(
        "build_transcript_history_body: {:?} ({} styled lines)",
        start.elapsed(),
        body.lines.len()
    );
    let start = Instant::now();
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, width);
    println!(
        "wrap_transcript: {:?} ({} visual rows)",
        start.elapsed(),
        wrapped.rows.len()
    );

    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    let start = Instant::now();
    terminal.draw(|frame| app.render(frame)).unwrap();
    println!("first full render (cache fill): {:?}", start.elapsed());

    time_frames("idle cached frames", &mut app, &mut terminal, 60, |_| {});
    time_frames("typing frames", &mut app, &mut terminal, 60, |app| {
        app.draft.push('x');
    });
    time_frames("scroll frames", &mut app, &mut terminal, 60, |app| {
        app.follow_output = false;
        app.transcript_scroll = app.transcript_scroll.saturating_add(3);
    });

    for (start, _) in app.step_folds(app.history.len()) {
        app.expanded_step_folds.insert(start);
    }
    app.bump_transcript_revision();
    let start = Instant::now();
    terminal.draw(|frame| app.render(frame)).unwrap();
    println!("first render with folds expanded: {:?}", start.elapsed());
    time_frames(
        "idle frames, folds expanded",
        &mut app,
        &mut terminal,
        30,
        |_| {},
    );
}

/// Live trailing `edit_file` batch with large args, unresolved while thinking.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_cursor_live_batch_frame() {
    for (batch, arg_kb) in [(4usize, 36usize), (4, 385), (9, 385)] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        seed_large_history(&mut app, 20);
        let payload =
            "fn resolve() { let probe = probe_provider(key); }\n".repeat(arg_kb * 1024 / 52);
        for i in 0..batch {
            app.apply_agent_tool_call(
                Some(format!("call_{i}")),
                "edit_file".to_string(),
                serde_json::json!({
                    "path": format!("src/services/router_{i}.rs"),
                    "old_string": payload,
                    "new_string": payload,
                }),
                vec![],
                None,
            );
        }
        app.sending = true;
        app.request_started_at = Some(Instant::now());
        app.pending_reasoning = "Let me look at these files. ".to_string();
        let mut terminal = Terminal::new(TestBackend::new(160, 50)).unwrap();
        let label = format!("cursor live batch ×{batch} @ {arg_kb}KB args, thinking + scroll");
        let mut i = 0usize;
        time_frames(&label, &mut app, &mut terminal, 60, |app| {
            i += 1;
            app.pending_reasoning
                .push_str(&format!("token{i} and more reasoning text flows here, "));
            app.tick_status_throttle();
            app.scroll_up_lines(3);
        });
    }
}

/// Typewriter frame: each tick reveals a slice of the buffered stream, so the
/// volatile tail re-renders — the cost that scales with reply length.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_typewriter_frame() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 20);
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    // A long reply already on screen, still typing out more — the worst case
    // for a per-tick tail re-render, which used to be O(reply) here.
    app.pending_response = (0..150).map(reply_block).collect::<Vec<_>>().join("\n\n");
    app.incoming_buffer = (150..200).map(reply_block).collect::<Vec<_>>().join("\n\n");
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    time_frames(
        "typewriter stream (74KB shown)",
        &mut app,
        &mut terminal,
        120,
        |app| {
            app.tick_typewriter();
        },
    );
}
