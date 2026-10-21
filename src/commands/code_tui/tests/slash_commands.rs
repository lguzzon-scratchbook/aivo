use super::super::*;
use super::helpers::*;

#[test]
fn test_parse_slash_command_with_argument() {
    assert_eq!(
        parse_slash_command("model claude-sonnet-4").unwrap(),
        SlashCommand::Model(Some("claude-sonnet-4".to_string()))
    );
    assert!(
        parse_slash_command("attach ./README.md").is_err(),
        "/attach was removed — @path in the message attaches instead"
    );
    assert_eq!(
        parse_slash_command("resume").unwrap(),
        SlashCommand::Resume(None)
    );
    assert!(
        parse_slash_command("detach 2").is_err(),
        "/detach was removed — deleting the inline tag detaches instead"
    );
    assert_eq!(
        parse_slash_command("mcp add fs npx -y srv").unwrap(),
        SlashCommand::Mcp(Some("add fs npx -y srv".to_string()))
    );
    assert_eq!(parse_slash_command("mcp").unwrap(), SlashCommand::Mcp(None));
}

#[test]
fn test_parse_slash_preview() {
    assert_eq!(
        parse_slash_command("preview assets/logo.svg").unwrap(),
        SlashCommand::Preview(Some("assets/logo.svg".to_string()))
    );
    assert_eq!(
        parse_slash_command("preview off").unwrap(),
        SlashCommand::Preview(Some("off".to_string()))
    );
    assert_eq!(
        parse_slash_command("preview").unwrap(),
        SlashCommand::Preview(None)
    );
}

#[test]
fn test_parse_slash_compact() {
    assert_eq!(
        parse_slash_command("compact").unwrap(),
        SlashCommand::Compact { fast: false }
    );
    assert_eq!(
        parse_slash_command("compact fast").unwrap(),
        SlashCommand::Compact { fast: true }
    );
    assert_eq!(
        parse_slash_command("compact now").unwrap(),
        SlashCommand::Compact { fast: false }
    );
}

#[test]
fn test_parse_slash_context() {
    assert_eq!(
        parse_slash_command("context").unwrap(),
        SlashCommand::Context
    );
}

#[test]
fn test_parse_slash_session() {
    assert_eq!(
        parse_slash_command("session").unwrap(),
        SlashCommand::Session
    );
}

/// `/context` always opens the breakdown, folding in the injected digest section when present.
#[tokio::test]
async fn test_context_overlay_shows_breakdown() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Nothing injected: the breakdown still opens.
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Context"), "title:\n{screen}");
    assert!(screen.contains("System prompt"), "segments:\n{screen}");
    assert!(screen.contains("Tools"), "segments:\n{screen}");
    assert!(
        !screen.contains("Injected context"),
        "no injection expected:\n{screen}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));

    // With an injected block, the breakdown also carries the injected section + body.
    app.injected_context = Some("# aivo context\n\n**Topic:** prior work".to_string());
    app.injected_context_summary = Some("injected ~9 tokens from claude session abc (2m)".into());
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 44);
    assert!(screen.contains("Injected context"), "section:\n{screen}");
    assert!(
        screen.contains("injected ~9 tokens from claude session abc"),
        "summary header:\n{screen}"
    );
    assert!(screen.contains("Topic:"), "body:\n{screen}");
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_compact_command_no_engine_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_compact_command(true).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("nothing to compact"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.agent_serve.is_none() && app.response_task.is_none());
}

#[test]
fn test_notice_spans_single_color() {
    let plain = (MUTED(), "just a status".to_string());
    let spans = notice_spans(Some(&plain)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].style.fg, Some(MUTED()));

    // ERROR keeps its `Error:` prefix and single span.
    let err = (ERROR(), "boom".to_string());
    let spans = notice_spans(Some(&err)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].content.as_ref(), "Error: boom");
}

#[test]
fn test_parse_slash_command_unknown() {
    let err = parse_slash_command("wat").unwrap_err().to_string();
    assert!(err.contains("Unknown command"), "{err}");
    assert!(err.contains("/help lists commands"), "{err}");
}

/// `/copy 0` is a usage error and out-of-range names the real count, instead of
/// claiming no reply exists.
#[tokio::test]
async fn test_copy_rejects_zero_and_reports_range() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "only reply".to_string(),
        reasoning_content: None,
        attachments: vec![],
        id: None,
        timestamp: None,
    });

    let err = app.copy_reply_to_clipboard(Some(0)).unwrap_err();
    assert!(err.to_string().contains("Usage"), "{err}");

    let err = app.copy_reply_to_clipboard(Some(5)).unwrap_err();
    assert!(err.to_string().contains("Only 1 reply"), "{err}");
}
