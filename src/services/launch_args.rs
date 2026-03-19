use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;

use crate::services::ai_launcher::AIToolType;
use crate::services::codex_model_map::map_model_for_codex_cli;

pub(crate) struct RuntimeArgs {
    pub(crate) args: Vec<String>,
    pub(crate) codex_model_catalog_path: Option<String>,
}

pub(crate) fn merge_preview_env(
    tool_env: &HashMap<String, String>,
    manual_env: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut merged = tool_env.clone();
    if let Some(manual) = manual_env {
        for (key, value) in manual {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

pub(crate) fn preview_args(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    env: &HashMap<String, String>,
) -> Vec<String> {
    let args = inject_claude_teammate_mode(tool, raw_args);
    if tool == AIToolType::Pi {
        return inject_pi_model(model, &args);
    }
    if tool != AIToolType::Codex {
        return args;
    }

    let use_responses_router = uses_responses_to_chat_router(env);
    let args = inject_codex_model(model, &args, use_responses_router);
    if should_preview_codex_model_catalog(model, use_responses_router) {
        let mut preview = vec![
            "--config".to_string(),
            "model_catalog_json=\"<temp:aivo-codex-model-catalog.json>\"".to_string(),
        ];
        preview.extend(args);
        return preview;
    }
    args
}

pub(crate) fn build_preview_notes(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    env: &HashMap<String, String>,
) -> Vec<String> {
    let mut notes = Vec::new();

    if tool == AIToolType::Claude
        && !raw_args
            .iter()
            .any(|arg| arg == "--teammate-mode" || arg.starts_with("--teammate-mode="))
    {
        notes.push("injects `--teammate-mode in-process` for Claude".to_string());
    }

    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_ROUTER"],
        "starts an Anthropic compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"],
        "starts an Anthropic-to-OpenAI compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_COPILOT_ROUTER"],
        "starts a Copilot router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_RESPONSES_TO_CHAT_ROUTER"],
        "starts a Responses-to-Chat router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER"],
        "starts a Copilot-backed Responses-to-Chat router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_GEMINI_ROUTER"],
        "starts a Gemini router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_GEMINI_COPILOT_ROUTER"],
        "starts a Copilot-backed Gemini router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_OPENCODE_ROUTER"],
        "starts an OpenCode compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_OPENCODE_COPILOT_ROUTER"],
        "starts a Copilot-backed OpenCode router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_SETUP_PI_AGENT_DIR"],
        "writes a temporary Pi agent dir with custom provider config",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_PI_COPILOT_ROUTER"],
        "starts a Copilot-backed Pi router on a random local port",
    );

    let use_responses_router = uses_responses_to_chat_router(env);
    if tool == AIToolType::Codex
        && model.is_some()
        && !raw_args.iter().any(|arg| {
            arg == "--model" || arg == "-m" || arg.starts_with("--model=") || arg.starts_with("-m=")
        })
    {
        notes.push("injects `-m <model>` for Codex".to_string());
    }
    if tool == AIToolType::Codex && should_preview_codex_model_catalog(model, use_responses_router)
    {
        notes.push("writes a temporary Codex model catalog file at launch time".to_string());
    }

    if tool == AIToolType::Pi
        && model.is_some()
        && !raw_args
            .iter()
            .any(|arg| arg == "--model" || arg.starts_with("--model="))
    {
        notes.push("injects `--model <model>` for Pi".to_string());
    }

    notes
}

pub(crate) async fn build_runtime_args(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    env: &HashMap<String, String>,
) -> Result<RuntimeArgs> {
    let args = inject_claude_teammate_mode(tool, raw_args);
    if tool == AIToolType::Pi {
        return Ok(RuntimeArgs {
            args: inject_pi_model(model, &args),
            codex_model_catalog_path: None,
        });
    }
    if tool != AIToolType::Codex {
        return Ok(RuntimeArgs {
            args,
            codex_model_catalog_path: None,
        });
    }

    let use_responses_router = uses_responses_to_chat_router(env);
    let codex_model_catalog_path =
        maybe_write_codex_model_catalog(model, use_responses_router).await?;
    let args = inject_codex_model(model, &args, use_responses_router);
    let args = inject_codex_model_catalog(codex_model_catalog_path.as_deref(), &args);

    Ok(RuntimeArgs {
        args,
        codex_model_catalog_path,
    })
}

fn uses_responses_to_chat_router(env: &HashMap<String, String>) -> bool {
    env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER")
        || env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER")
}

fn maybe_push_router_note(
    notes: &mut Vec<String>,
    env: &HashMap<String, String>,
    env_keys: &[&str],
    note: &str,
) {
    if env_keys.iter().any(|key| env.contains_key(*key)) {
        notes.push(note.to_string());
    }
}

fn should_preview_codex_model_catalog(model: Option<&str>, uses_non_openai_router: bool) -> bool {
    let model = match model {
        Some(model) if !model.is_empty() => model,
        _ => return false,
    };

    if !uses_non_openai_router {
        return false;
    }

    let model_lower = model.to_lowercase();
    let name_only = model_lower.split('/').next_back().unwrap_or(&model_lower);
    !(name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4"))
}

fn inject_claude_teammate_mode(tool: AIToolType, args: &[String]) -> Vec<String> {
    if tool != AIToolType::Claude {
        return args.to_vec();
    }

    let has_teammate_mode = args
        .iter()
        .any(|a| a == "--teammate-mode" || a.starts_with("--teammate-mode="));
    if has_teammate_mode {
        return args.to_vec();
    }

    let mut new_args = vec!["--teammate-mode".to_string(), "in-process".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

fn inject_pi_model(model: Option<&str>, args: &[String]) -> Vec<String> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return args.to_vec(),
    };

    let has_model_flag = args
        .iter()
        .any(|a| a == "--model" || a.starts_with("--model="));
    if has_model_flag {
        return args.to_vec();
    }

    // Always prefix model with "aivo/" so pi selects
    // the custom provider from models.json.
    let pi_model = format!("aivo/{model}");

    let mut new_args = vec!["--model".to_string(), pi_model];
    new_args.extend_from_slice(args);
    new_args
}

fn inject_codex_model(model: Option<&str>, args: &[String], use_router: bool) -> Vec<String> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return args.to_vec(),
    };

    let has_model_flag = args
        .iter()
        .any(|a| a == "--model" || a == "-m" || a.starts_with("--model=") || a.starts_with("-m="));
    if has_model_flag {
        return args.to_vec();
    }

    let codex_model = if use_router {
        model.to_string()
    } else {
        map_model_for_codex_cli(model)
    };
    let mut new_args = vec!["-m".to_string(), codex_model];
    new_args.extend_from_slice(args);
    new_args
}

fn inject_codex_model_catalog(path: Option<&str>, args: &[String]) -> Vec<String> {
    let path = match path {
        Some(p) if !p.is_empty() => p,
        _ => return args.to_vec(),
    };

    if args.iter().any(|a| a.contains("model_catalog_json")) {
        return args.to_vec();
    }

    let escaped_path = path.replace('\\', "\\\\").replace('"', "\\\"");
    let mut new_args = vec![
        "--config".to_string(),
        format!("model_catalog_json=\"{}\"", escaped_path),
    ];
    new_args.extend_from_slice(args);
    new_args
}

async fn maybe_write_codex_model_catalog(
    model: Option<&str>,
    uses_non_openai_router: bool,
) -> Result<Option<String>> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return Ok(None),
    };

    if !uses_non_openai_router {
        return Ok(None);
    }

    let model_lower = model.to_lowercase();
    let name_only = model_lower.split('/').next_back().unwrap_or(&model_lower);
    if name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
    {
        return Ok(None);
    }

    let catalog_json = build_codex_model_catalog_json(model)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = format!(
        "aivo-codex-model-catalog-{}-{}.json",
        std::process::id(),
        nonce
    );
    let path = std::env::temp_dir().join(file_name);

    tokio::fs::write(&path, catalog_json)
        .await
        .with_context(|| {
            format!(
                "Failed to write Codex model catalog override at {}",
                path.display()
            )
        })?;

    Ok(Some(path.to_string_lossy().to_string()))
}

fn build_codex_model_catalog_json(model: &str) -> Result<String> {
    let catalog = json!({
        "models": [{
            "slug": model,
            "display_name": model,
            "description": format!("Custom model metadata for {}", model),
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "minimal_client_version": [0, 1, 0],
            "supported_in_api": true,
            "priority": 0,
            "upgrade": serde_json::Value::Null,
            "base_instructions": "base instructions",
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": serde_json::Value::Null,
            "apply_patch_tool_type": serde_json::Value::Null,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": false,
            "supports_image_detail_original": false,
            "context_window": 272000,
            "experimental_supported_tools": []
        }]
    });
    Ok(serde_json::to_string(&catalog)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_claude_teammate_mode_for_claude() {
        let args = vec!["--verbose".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(
            result,
            vec!["--teammate-mode", "in-process", "--verbose", "prompt"]
        );
    }

    #[test]
    fn test_inject_claude_teammate_mode_skips_non_claude() {
        let args = vec!["--verbose".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Codex, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Gemini, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Opencode, &args);
        assert_eq!(result, vec!["--verbose"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_respects_user_flag() {
        let args = vec![
            "--teammate-mode".to_string(),
            "split".to_string(),
            "prompt".to_string(),
        ];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "split", "prompt"]);

        let args = vec!["--teammate-mode=split".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode=split", "prompt"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_empty_args() {
        let args: Vec<String> = vec![];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "in-process"]);
    }

    #[test]
    fn test_inject_codex_model_injects_when_provided() {
        let model = Some("o4-mini");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "o4-mini", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_original() {
        let model = Some("kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_namespaced() {
        let model = Some("moonshot/kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "moonshot/kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_when_already_specified() {
        let model = Some("o4-mini");
        let args = vec![
            "--model".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["--model", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_shorthand_flag() {
        let model = Some("o4-mini");
        let args = vec![
            "-m".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_equals_format() {
        let model = Some("o4-mini");
        let args = vec!["--model=gpt-4o".to_string(), "file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["--model=gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_empty_model() {
        let model = Some("");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_none_model() {
        let model: Option<&str> = None;
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_catalog_injects_when_path_provided() {
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(
            result,
            vec![
                "--config",
                "model_catalog_json=\"/tmp/catalog.json\"",
                "file.ts"
            ]
        );
    }

    #[test]
    fn test_inject_codex_model_catalog_skips_when_existing_setting_present() {
        let args = vec![
            "--config".to_string(),
            "model_catalog_json=\"/tmp/custom.json\"".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(result, args);
    }

    #[test]
    fn test_build_codex_model_catalog_json_includes_model_slug() {
        let model = "minimax/minimax-m2.5";
        let json = build_codex_model_catalog_json(model).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["models"][0]["slug"], model);
        assert_eq!(parsed["models"][0]["display_name"], model);
    }
}
