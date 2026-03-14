pub fn map_model_for_codex_cli(model: &str) -> String {
    let model_lower = model.to_lowercase();

    if model_lower.starts_with("gpt-")
        || model_lower.starts_with("o1")
        || model_lower.starts_with("o3")
        || model_lower.starts_with("o4")
        || model_lower.starts_with("chatgpt")
    {
        return model.to_string();
    }

    let name_only = model_lower.split('/').next_back().unwrap_or(&model_lower);

    let is_high_capability = name_only.contains("opus")
        || name_only.contains("405b")
        || name_only.contains("r1")
        || name_only.contains("reasoner")
        || name_only.contains("k2.5")
        || name_only.contains("k2-5")
        || name_only.contains("large")
        || name_only.contains("pro");

    let is_lightweight = name_only.contains("flash")
        || name_only.contains("haiku")
        || name_only.contains("small")
        || name_only.contains("mini")
        || name_only.contains("8b")
        || name_only.contains("11b");

    if is_high_capability {
        if name_only.contains("reasoner") || name_only.contains("r1") {
            "o1".to_string()
        } else {
            "gpt-4o".to_string()
        }
    } else if is_lightweight {
        "gpt-4o-mini".to_string()
    } else {
        "gpt-4o".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::map_model_for_codex_cli;

    #[test]
    fn keeps_known_openai_models() {
        assert_eq!(map_model_for_codex_cli("gpt-4o"), "gpt-4o");
        assert_eq!(map_model_for_codex_cli("o3"), "o3");
        assert_eq!(
            map_model_for_codex_cli("chatgpt-4o-latest"),
            "chatgpt-4o-latest"
        );
    }

    #[test]
    fn maps_reasoning_and_large_models() {
        assert_eq!(map_model_for_codex_cli("deepseek/deepseek-r1"), "o1");
        assert_eq!(map_model_for_codex_cli("anthropic/claude-opus-4"), "gpt-4o");
    }

    #[test]
    fn maps_lightweight_models() {
        assert_eq!(map_model_for_codex_cli("gemini-2.0-flash"), "gpt-4o-mini");
        assert_eq!(map_model_for_codex_cli("meta/llama-3.1-8b"), "gpt-4o-mini");
    }
}
