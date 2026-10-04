use super::super::*;
use super::helpers::*;
use serde_json::json;

fn request_tool_names(engine: &AgentEngine) -> Vec<String> {
    engine
        .request_tools()
        .iter()
        .map(|t| {
            t["function"]["name"]
                .as_str()
                .or(t["type"].as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

fn request_system(engine: &AgentEngine) -> String {
    engine.outgoing_messages()[0]["content"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn ask_mode_request_carries_lean_prompt_and_web_tools_only() {
    let dir = tmp();
    std::fs::write(dir.join("AGENTS.md"), "Always use tabs.").unwrap();
    let cwd = dir.display().to_string();
    let mut engine = AgentEngine::new(
        &cwd,
        "m",
        "2026-10-04",
        &["AGENTS.md".to_string()],
        &[],
        0,
        0,
    );
    engine.append_system_context("<context>project notes</context>");
    let stored = system_content(&engine);
    assert!(
        stored.contains("Always use tabs."),
        "guide inlined normally"
    );

    engine.set_ask_mode(true);
    assert!(engine.read_only);
    let sys = request_system(&engine);
    assert!(sys.contains("research and learning assistant"), "{sys}");
    assert!(sys.contains("Current date: 2026-10-04."));
    for absent in [
        cwd.as_str(),
        "Always use tabs.",
        "project notes",
        "run_bash",
    ] {
        assert!(!sys.contains(absent), "{absent:?} leaked into ask prompt");
    }
    let mut names = request_tool_names(&engine);
    names.sort();
    assert_eq!(names, ["web_fetch", "web_search"]);
    assert_eq!(system_content(&engine), stored, "stored prompt untouched");
}

#[test]
fn ask_mode_keeps_native_search_tool() {
    let mut engine = AgentEngine::new("/tmp", "claude-sonnet-5", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    let mut names = request_tool_names(&engine);
    names.sort();
    assert_eq!(names, ["web_fetch", "web_search"]);
    assert!(
        engine
            .request_tools()
            .iter()
            .any(|t| t["type"].as_str() == Some("web_search"))
    );
}

#[test]
fn ask_prompt_says_when_web_search_is_off() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.set_web_search_enabled(false);
    engine.set_ask_mode(true);
    assert_eq!(request_tool_names(&engine), ["web_fetch"]);
    assert!(request_system(&engine).contains("web search is off"));
}

#[test]
fn set_ask_mode_round_trips_without_touching_tools_or_prompt() {
    for model in ["m", "gpt-5"] {
        let mut engine = AgentEngine::new("/tmp", model, "", &[], &[], 0, 0);
        let before_tools = tool_names(&engine);
        let before_system = system_content(&engine);

        engine.set_ask_mode(true);
        engine.set_ask_mode(true); // idempotent on
        assert_eq!(tool_names(&engine), before_tools, "model={model}");
        engine.set_ask_mode(false);
        engine.set_ask_mode(false); // idempotent off

        assert!(!engine.read_only, "model={model}");
        assert_eq!(tool_names(&engine), before_tools, "model={model}");
        assert_eq!(system_content(&engine), before_system, "model={model}");
        assert_eq!(request_system(&engine), before_system, "model={model}");
    }
}

#[test]
fn plan_and_ask_mode_are_exclusive_and_swap_cleanly() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let before_tools = tool_names(&engine);
    let before_system = system_content(&engine);

    engine.set_plan_mode(true);
    engine.set_ask_mode(true); // leaves plan mode first
    assert!(!system_content(&engine).contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
    assert!(!tool_names(&engine).iter().any(|n| n == "exit_plan_mode"));
    assert!(request_system(&engine).contains("research and learning assistant"));

    engine.set_plan_mode(true); // leaves ask mode first
    let sys = request_system(&engine);
    assert!(sys.contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
    assert!(!sys.contains("research and learning assistant"));
    assert!(tool_names(&engine).iter().any(|n| n == "exit_plan_mode"));

    engine.set_plan_mode(false);
    assert!(!engine.read_only);
    let mut after = tool_names(&engine);
    let mut before = before_tools.clone();
    after.sort();
    before.sort();
    assert_eq!(after, before, "tool set restored exactly");
    assert_eq!(
        system_content(&engine),
        before_system,
        "no directive residue"
    );
}

#[test]
fn image_model_set_during_ask_mode_stays_off_the_wire() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    engine.set_image_source(Some(test_generator()));
    assert!(
        !request_tool_names(&engine)
            .iter()
            .any(|n| n == "generate_image")
    );
    engine.set_ask_mode(false);
    assert_eq!(
        request_tool_names(&engine)
            .iter()
            .filter(|n| *n == "generate_image")
            .count(),
        1
    );
}

#[tokio::test]
async fn ask_mode_refuses_local_tools() {
    let dir = tmp();
    let batch = batch_tool_call_sse(&[
        (
            "c1",
            "write_file",
            json!({"path": "out.txt", "content": "hi"}),
        ),
        ("c2", "run_bash", json!({"command": "echo probe"})),
        ("c3", "read_file", json!({"path": "Cargo.toml"})),
    ]);
    let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("explain it".into()),
        &mut ui,
    )
    .await;

    let results = tool_result_texts(&engine);
    let refused = results
        .iter()
        .filter(|c| c.contains("Ask mode has no local tools"))
        .count();
    assert_eq!(refused, 3, "{results:?}");
    assert!(
        !results.iter().any(|c| c.contains("probe")),
        "bash never ran"
    );
    assert!(!dir.join("out.txt").exists(), "no file written in ask mode");
    assert!(
        ui.ask_tools.is_empty(),
        "refused without a permission prompt"
    );
    assert!(engine.read_only, "ask mode persists through the turn");
}

/// A hallucinated `exit_plan_mode` must not pop the approval card (the UI would
/// approve) or drop the read-only floor.
#[tokio::test]
async fn ask_mode_refuses_exit_plan_mode() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. do it"}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Approve),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("explain".into()),
        &mut ui,
    )
    .await;

    let results = tool_result_texts(&engine);
    assert!(
        results
            .iter()
            .any(|c| c.contains("Ask mode has no local tools")),
        "exit_plan_mode refused: {results:?}"
    );
    assert!(engine.read_only, "ask mode's read-only floor holds");
}

#[test]
fn outgoing_reminder_only_in_plan_mode() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.push_text_turn("user", "hello".to_string());

    engine.set_ask_mode(true);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(!tail.contains("still active"), "ask mode adds no reminder");

    engine.set_plan_mode(true);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(tail.contains("Plan mode is still active"));

    engine.set_plan_mode(false);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(!tail.contains("still active"), "no reminder outside a mode");
}

#[test]
fn context_report_reflects_the_ask_request() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let normal = engine.context_report();
    engine.set_ask_mode(true);
    let ask = engine.context_report();
    assert!(ask.system_prompt < normal.system_prompt);
    assert!(ask.tools < normal.tools);
    assert_eq!(ask.tool_count, 2);
}
