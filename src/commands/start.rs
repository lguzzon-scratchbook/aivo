use anyhow::Result;
use console::{Key, Term};

use crate::cli::parse_env_vars;
use crate::commands::keys::prompt_pick_key_without_activation;
use crate::commands::models::fetch_models_for_select;
use crate::commands::print_launch_preview;
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::http_utils;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, DirectoryStartRecord, SessionStore};
use crate::services::system_env;
use crate::style;
use crate::tui::FuzzySelect;

#[derive(Debug, Clone)]
pub struct StartFlowArgs {
    pub model: Option<String>,
    pub key: Option<String>,
    pub tool: Option<String>,
    pub debug: bool,
    pub dry_run: bool,
    pub yes: bool,
    pub envs: Vec<String>,
}

struct Resolved<T> {
    value: T,
    interactive: bool,
}

pub struct StartCommand {
    session_store: SessionStore,
    ai_launcher: AILauncher,
    cache: ModelsCache,
}

impl StartCommand {
    pub fn new(session_store: SessionStore, ai_launcher: AILauncher, cache: ModelsCache) -> Self {
        Self {
            session_store,
            ai_launcher,
            cache,
        }
    }

    pub async fn execute(&self, args: StartFlowArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, args: StartFlowArgs) -> Result<ExitCode> {
        let cwd = system_env::current_dir_string()
            .ok_or_else(|| anyhow::anyhow!("Failed to determine the current directory"))?;
        let remembered = self.session_store.get_directory_start(&cwd).await?;

        if remembered.is_none() {
            eprintln!(
                "{}",
                style::dim("No saved start record for this directory yet. I’ll help you pick one.")
            );
        }

        let key = self
            .resolve_key(args.key.as_deref(), remembered.as_ref())
            .await?;
        let tool = self.resolve_tool(args.tool.as_deref(), remembered.as_ref())?;
        let model = self
            .resolve_model(args.model, remembered.as_ref(), &key)
            .await?;
        let env = parse_env_vars(&args.envs);
        let skip_confirm =
            remembered.is_some() || (key.interactive && tool.interactive && model.interactive);

        if args.dry_run {
            let plan = self
                .ai_launcher
                .prepare_launch(&LaunchOptions {
                    tool: tool.value,
                    args: Vec::new(),
                    debug: args.debug,
                    model: model.value,
                    env: (!env.is_empty()).then_some(env),
                    key_override: Some(key.value),
                })
                .await?;
            print_launch_preview(&plan);
            return Ok(ExitCode::Success);
        }

        if !args.yes {
            let provider = normalize_provider_label(&key.value.base_url);
            eprintln!(
                "{}{}",
                style::cyan(tool.value.as_str()),
                style::dim(format!(
                    " · {} · {}",
                    provider,
                    model.value.as_deref().unwrap_or("(tool default)")
                ))
            );
            if !skip_confirm && !confirm("Run?")? {
                return Ok(ExitCode::Success);
            }
        }

        let exit_code = self
            .ai_launcher
            .launch(&LaunchOptions {
                tool: tool.value,
                args: Vec::new(),
                debug: args.debug,
                model: model.value,
                env: (!env.is_empty()).then_some(env),
                key_override: Some(key.value),
            })
            .await?;

        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    async fn resolve_key(
        &self,
        key_arg: Option<&str>,
        remembered: Option<&DirectoryStartRecord>,
    ) -> Result<Resolved<ApiKey>> {
        if matches!(key_arg, Some("")) {
            return self.prompt_select_key(remembered).await;
        }

        if let Some(key_id_or_name) = key_arg {
            return Ok(Resolved {
                value: self
                    .session_store
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await?,
                interactive: false,
            });
        }

        if let Some(record) = remembered
            && let Some(key) = self.session_store.get_key_by_id(&record.key_id).await?
        {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        if let Some(key) = self.session_store.get_active_key().await? {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => anyhow::bail!("No API key configured. Run 'aivo keys add' first."),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => match prompt_pick_key_without_activation(&keys, "Select key", 0)? {
                Some(key) => Ok(Resolved {
                    value: key,
                    interactive: true,
                }),
                None => Err(anyhow::anyhow!("Cancelled")),
            },
        }
    }

    async fn prompt_select_key(
        &self,
        remembered: Option<&DirectoryStartRecord>,
    ) -> Result<Resolved<ApiKey>> {
        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => anyhow::bail!("No API key configured. Run 'aivo keys add' first."),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => {
                let active_key_id = self
                    .session_store
                    .get_active_key_info()
                    .await?
                    .map(|active_key| active_key.id);
                let default_idx = remembered
                    .and_then(|record| keys.iter().position(|key| key.id == record.key_id))
                    .or_else(|| {
                        active_key_id
                            .as_ref()
                            .and_then(|active_id| keys.iter().position(|key| &key.id == active_id))
                    })
                    .unwrap_or(0);
                match prompt_pick_key_without_activation(&keys, "Select key", default_idx)? {
                    Some(key) => Ok(Resolved {
                        value: key,
                        interactive: true,
                    }),
                    None => Err(anyhow::anyhow!("Cancelled")),
                }
            }
        }
    }

    fn resolve_tool(
        &self,
        tool_arg: Option<&str>,
        remembered: Option<&DirectoryStartRecord>,
    ) -> Result<Resolved<AIToolType>> {
        if let Some(tool) = tool_arg {
            return Ok(Resolved {
                value: AIToolType::parse(tool)
                    .ok_or_else(|| anyhow::anyhow!("Unknown AI tool '{}'", tool))?,
                interactive: false,
            });
        }

        if let Some(record) = remembered
            && let Some(tool) = AIToolType::parse(&record.tool)
        {
            return Ok(Resolved {
                value: tool,
                interactive: false,
            });
        }

        let tools = AIToolType::all();
        let items = tools
            .iter()
            .map(|t| t.as_str().to_string())
            .collect::<Vec<_>>();
        let selected = FuzzySelect::new()
            .with_prompt("Select tool")
            .items(&items)
            .default(0)
            .interact_opt()
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("Cancelled"))?;
        Ok(Resolved {
            value: tools[selected],
            interactive: true,
        })
    }

    async fn resolve_model(
        &self,
        model_arg: Option<String>,
        remembered: Option<&DirectoryStartRecord>,
        key: &Resolved<ApiKey>,
    ) -> Result<Resolved<Option<String>>> {
        let should_prompt = model_arg.as_ref().is_some_and(|value| value.is_empty())
            || (model_arg.is_none() && remembered.is_none());

        if should_prompt {
            return self.prompt_select_model(&key.value).await;
        }

        match model_arg {
            Some(value) => Ok(Resolved {
                value: Some(value),
                interactive: false,
            }),
            None => Ok(Resolved {
                value: remembered.and_then(|record| record.model.clone()),
                interactive: false,
            }),
        }
    }

    async fn prompt_select_model(&self, key: &ApiKey) -> Result<Resolved<Option<String>>> {
        let client = http_utils::router_http_client();
        let models = fetch_models_for_select(&client, key, &self.cache).await;
        if models.is_empty() {
            anyhow::bail!(
                "No models found for this key. Use 'aivo models --refresh' or specify one with --model <name>."
            );
        }
        let selected = FuzzySelect::new()
            .with_prompt("Select model")
            .items(&models)
            .default(0)
            .interact_opt()
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("Cancelled"))?;
        Ok(Resolved {
            value: Some(models[selected].clone()),
            interactive: true,
        })
    }
}

fn confirm(prompt: &str) -> std::io::Result<bool> {
    let term = Term::stdout();
    term.write_str(&format!("{prompt} [Y/n] "))?;

    loop {
        match term.read_key()? {
            Key::Enter | Key::Char('y') | Key::Char('Y') => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Running..."))?;
                return Ok(true);
            }
            Key::Char('n') | Key::Char('N') | Key::Escape => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Cancelled."))?;
                return Ok(false);
            }
            _ => {}
        }
    }
}

fn normalize_provider_label(base_url: &str) -> String {
    if base_url == "copilot" {
        return "github.com/copilot".to_string();
    }
    if base_url == "ollama" {
        return "localhost/ollama".to_string();
    }

    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(ToString::to_string))
        .unwrap_or_else(|| base_url.to_string())
}
