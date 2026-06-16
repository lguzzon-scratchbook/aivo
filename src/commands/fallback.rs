/**
 * FallbackCommand handler — manage fallback definitions.
 *
 * Fallback definitions map short names (e.g. "auto") to ordered lists of
 * provider/model targets. When a fallback alias is used as --model, the
 * system tries each target in sequence until one succeeds.
 */
use anyhow::Result;

use crate::cli::FallbackArgs;
use crate::errors::ExitCode;
use crate::services::fallback::{
    Entry, FallbackDefinition, FallbackReference, ProviderModelPair, validate_fallback_registry,
};
use crate::services::session_store::SessionStore;
use crate::style;

pub struct FallbackCommand {
    session_store: SessionStore,
}

impl FallbackCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: FallbackArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, args: FallbackArgs) -> Result<ExitCode> {
        // `aivo fallback --rm <name>`
        if let Some(ref name) = args.rm {
            if args.json {
                anyhow::bail!("--json only applies to listing fallbacks");
            }
            return self.remove_fallback(name).await;
        }

        // `aivo fallback --set <name> -- <targets>`
        if let Some(ref name) = args.set {
            if args.json {
                anyhow::bail!("--json only applies to listing fallbacks");
            }
            return self.set_fallback(name, &args.targets).await;
        }

        // `aivo fallback` — list all
        self.list_fallbacks(args.json).await
    }

    async fn list_fallbacks(&self, json: bool) -> Result<ExitCode> {
        let fallbacks = self.session_store.get_fallbacks().await?;

        if json {
            let payload: serde_json::Map<String, serde_json::Value> = fallbacks
                .iter()
                .map(|(id, def)| {
                    let entries: Vec<serde_json::Value> = def
                        .sequence
                        .iter()
                        .map(|e| match e {
                            Entry::ProviderModelPair(pmp) => {
                                serde_json::json!({ "provider": pmp.provider, "model": pmp.model })
                            }
                            Entry::FallbackReference(fr) => {
                                serde_json::json!({ "fallbackId": fr.fallback_id })
                            }
                        })
                        .collect();
                    let obj = serde_json::json!({
                        "description": def.description,
                        "timeoutMs": def.timeout_ms,
                        "sequence": entries,
                    });
                    (id.clone(), obj)
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if fallbacks.is_empty() {
            println!("{}", style::dim("No fallback definitions."));
            println!();
            println!(
                "{}",
                style::dim(
                    "Create one with: aivo fallback --set auto -- anthropic:claude-sonnet-4-6 openai:gpt-4o"
                )
            );
            return Ok(ExitCode::Success);
        }

        let mut entries: Vec<_> = fallbacks.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        for (id, def) in &entries {
            let desc = def
                .description
                .as_deref()
                .map(|d| format!(" — {}", d))
                .unwrap_or_default();
            println!("{} {}{}", style::cyan(id), style::bold(" (fallback)"), desc);

            for entry in &def.sequence {
                match entry {
                    Entry::ProviderModelPair(pmp) => {
                        println!(
                            "  {} {}:{}",
                            style::arrow_symbol(),
                            style::dim(&pmp.provider),
                            style::dim(&pmp.model),
                        );
                    }
                    Entry::FallbackReference(fr) => {
                        println!(
                            "  {} {} {}",
                            style::arrow_symbol(),
                            style::dim("→"),
                            style::cyan(&fr.fallback_id),
                        );
                    }
                }
            }

            if let Some(timeout) = def.timeout_ms {
                println!(
                    "  {} {}",
                    style::dim("timeout:"),
                    style::dim(format!("{}ms", timeout)),
                );
            }
            println!();
        }

        Ok(ExitCode::Success)
    }

    async fn set_fallback(&self, name: &str, targets: &[String]) -> Result<ExitCode> {
        if name.is_empty() {
            anyhow::bail!("Fallback name cannot be empty");
        }

        if targets.is_empty() {
            anyhow::bail!(
                "No targets specified. Usage: aivo fallback --set <name> -- <provider:model>..."
            );
        }

        // Parse targets: "provider:model" or "@fallback_id"
        let mut sequence = Vec::new();
        for target in targets {
            if let Some(fb_id) = target.strip_prefix('@') {
                if fb_id.is_empty() {
                    anyhow::bail!("Empty fallback reference in target '{}'", target);
                }
                sequence.push(Entry::FallbackReference(FallbackReference {
                    fallback_id: fb_id.to_string(),
                }));
            } else if let Some((provider, model)) = target.split_once(':') {
                if provider.is_empty() || model.is_empty() {
                    anyhow::bail!(
                        "Invalid target '{}'. Use format 'provider:model' or '@fallback_id'",
                        target
                    );
                }
                sequence.push(Entry::ProviderModelPair(ProviderModelPair {
                    provider: provider.to_string(),
                    model: model.to_string(),
                }));
            } else {
                anyhow::bail!(
                    "Invalid target '{}'. Use format 'provider:model' (e.g. anthropic:claude-sonnet-4-6) or '@fallback_id'",
                    target
                );
            }
        }

        let def = FallbackDefinition {
            id: name.to_string(),
            description: None,
            timeout_ms: None,
            sequence,
        };

        // Validate the new definition in the context of existing fallbacks
        let mut all = self.session_store.get_fallbacks().await?;
        all.insert(name.to_string(), def);

        validate_fallback_registry(&all).map_err(|errors| {
            anyhow::anyhow!(
                "Validation failed:\n{}",
                errors
                    .iter()
                    .map(|e| format!("  - {}", e.message))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })?;

        // Persist
        self.session_store.set_fallback(name, &all).await?;

        println!(
            "{} Set fallback '{}' with {} target(s)",
            style::green("✓"),
            style::cyan(name),
            targets.len(),
        );
        Ok(ExitCode::Success)
    }

    async fn remove_fallback(&self, name: &str) -> Result<ExitCode> {
        let mut all = self.session_store.get_fallbacks().await?;
        if all.remove(name).is_none() {
            anyhow::bail!("Fallback '{}' not found", name);
        }
        self.session_store.set_fallback(name, &all).await?;
        println!(
            "{} Removed fallback '{}'",
            style::green("✓"),
            style::cyan(name)
        );
        Ok(ExitCode::Success)
    }
}
