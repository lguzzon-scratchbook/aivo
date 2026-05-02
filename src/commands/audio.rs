//! `aivo audio` — generate speech (TTS) from a text prompt.
//!
//! Resolves a key, takes the prompt from the positional arg, calls the
//! provider, saves the result. Mirrors `aivo image`'s flow exactly: the
//! shared `services::audio_gen` module handles HTTP + bytes; this module
//! owns the CLI ergonomics (help text, model picker, overwrite policy,
//! human / JSON output). When no prompt is provided we print the command
//! help and the audio-scope active key/model.

use std::fs;
use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::json;
use tokio::task::JoinHandle;

use crate::cli::AudioArgs;
use crate::errors::ExitCode;
use crate::services::audio_gen::{self, AudioArtifact, AudioRequest};
use crate::services::http_utils::router_http_client;
use crate::services::media_io::{self, OutputTarget, OverwritePolicy, human_bytes};
use crate::services::models_cache::ModelsCache;
use crate::services::playback;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct AudioCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl AudioCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub fn print_help() {
        Self::print_help_for("aivo audio", "— generate speech (TTS) from a prompt", false);
    }

    /// Same help structure as `aivo audio`, but tweaks the description and
    /// examples to highlight the play-by-default behavior.
    pub fn print_speak_help() {
        Self::print_help_for("aivo speak", "— speak a prompt aloud (TTS + play)", true);
    }

    fn print_help_for(name: &str, blurb: &str, default_play: bool) {
        println!("{} {}", style::cyan(name), style::dim(blurb));
        println!();
        println!("{} {} [OPTIONS] <PROMPT>", style::bold("Usage:"), name);
        println!();
        println!("{}", style::bold("Arguments:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<24}", "PROMPT")),
            style::dim("Text to read aloud")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let opt = |f: &str, d: &str| {
            println!("  {}{}", style::cyan(format!("{:<24}", f)), style::dim(d));
        };
        opt("-m, --model <MODEL>", "TTS model (e.g. tts-1, tts-1-hd)");
        opt("-k, --key <ID|NAME>", "API key to use");
        opt(
            "-o, --output <PATH>",
            "File, directory, or template ({ts}/{model})",
        );
        opt("-f, --force", "Overwrite existing files without prompting");
        opt("    --voice <VOICE>", "alloy | nova | onyx | echo | …");
        opt(
            "    --format <FORMAT>",
            "mp3 (default) | wav | opus | aac | flac",
        );
        opt("    --speed <SPEED>", "Playback speed, typically 0.25–4.0");
        if default_play {
            opt(
                "    --no-play",
                "Save without playing (default for `aivo audio`)",
            );
        } else {
            opt(
                "    --play",
                "Play through speakers after generation (or use `aivo speak`)",
            );
        }
        opt("-r, --refresh", "Bypass model-list cache");
        opt("    --json", "Emit JSON result (for scripting)");
        println!();
        println!("{}", style::bold("Examples:"));
        if default_play {
            println!("  {}", style::dim("aivo speak \"hello world\""));
            println!(
                "  {}",
                style::dim("aivo speak \"narration line\" -m tts-1-hd --voice nova")
            );
            println!(
                "  {}",
                style::dim("aivo speak \"...\" --no-play -o out.mp3   # save only")
            );
        } else {
            println!("  {}", style::dim("aivo audio \"hello world\""));
            println!(
                "  {}",
                style::dim("aivo audio \"narration line\" -m tts-1-hd --voice nova -o out.mp3")
            );
            println!(
                "  {}",
                style::dim("aivo audio \"...\" --play   # save mp3 and play it")
            );
        }
    }

    /// Prints the audio-scope active key and model under the help output.
    /// Reads the audio-only `last_audio_selection` slot so it doesn't surface
    /// a chat key the user picked for `aivo chat`.
    pub async fn print_active_selection(session_store: &SessionStore) {
        let sel = session_store
            .get_last_audio_selection()
            .await
            .ok()
            .flatten();
        crate::commands::print_active_selection_for(session_store, sel).await;
    }

    /// `default_play` flips the play-vs-save default. `aivo audio` passes
    /// `false` (save unless `--play`); `aivo speak` passes `true` (play
    /// unless `--no-play`).
    pub async fn execute(self, args: AudioArgs, key: ApiKey, default_play: bool) -> ExitCode {
        // No prompt → print help + audio-scope active selection. Don't fall
        // back to stdin: a misfired empty stdin shouldn't fire a model picker
        // and burn an API call.
        let prompt = match args
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(p) => p.to_string(),
            None => {
                if default_play {
                    Self::print_speak_help();
                } else {
                    Self::print_help();
                }
                Self::print_active_selection(&self.session_store).await;
                return ExitCode::Success;
            }
        };

        let model = match resolve_audio_model(&self.session_store, &self.cache, &args, &key).await {
            Ok(Some(m)) => m,
            Ok(None) => return ExitCode::Success, // picker cancelled
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                return ExitCode::UserError;
            }
        };

        let play_requested = args.play || (default_play && !args.no_play);
        let wants_save = args.output.is_some();
        // Pure-play mode: ask for raw PCM. It's the only format every
        // OpenAI TTS model accepts (gpt-4o-mini-tts rejects "wav" as
        // `Invalid option: expected one of "mp3"|"pcm"`), and Gemini
        // returns L16 PCM regardless of what we ask for. `audio_gen`
        // wraps the raw bytes in a WAV header before we save them, so
        // afplay/aplay/SoundPlayer get a playable file on every platform.
        // When -o is set we honor the user's --format and let playback
        // be best-effort against the saved file.
        let request_format = if play_requested && !wants_save {
            Some("pcm".to_string())
        } else {
            args.format.clone()
        };

        let (initial_path, is_temp) = if play_requested && !wants_save {
            let suffix: u32 = rand::random();
            let path = std::env::temp_dir().join(format!("aivo-tts-{suffix:x}.wav"));
            (path, true)
        } else {
            let target = OutputTarget::parse(args.output.as_deref());
            let ext = default_extension(request_format.as_deref());
            let initial = match media_io::resolve_output_path(&target, &model, &ext) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    return ExitCode::UserError;
                }
            };
            let policy = OverwritePolicy::from_flags(args.force, args.json);
            let resolved = match crate::commands::resolve_final_path(&initial, policy) {
                Some(p) => p,
                None => return ExitCode::UserError,
            };
            (resolved, false)
        };

        // For temp files we picked the `.wav` extension ourselves and the
        // caller will play whatever the provider returns; treat as pinned so
        // the server can't silently rename the temp path.
        let pinned = if is_temp {
            true
        } else {
            OutputTarget::parse(args.output.as_deref()).pins_extension()
        };

        let request = AudioRequest {
            prompt,
            model: model.clone(),
            voice: args.voice.clone(),
            format: request_format.clone(),
            speed: args.speed,
        };

        let spinner = start_spinner_if_tty(&model, play_requested && !wants_save);
        let start = std::time::Instant::now();
        let result = audio_gen::generate(&key, &request, Some(&initial_path), pinned).await;
        let elapsed = start.elapsed();
        stop_spinner(spinner);

        let artifact = match result {
            Ok(a) => a,
            Err(e) => {
                if is_temp {
                    let _ = fs::remove_file(&initial_path);
                }
                eprintln!("{} {}", style::red("Error:"), e);
                return ExitCode::NetworkError;
            }
        };

        let mut played = false;
        let mut playback_error: Option<String> = None;
        if play_requested {
            let target_for_play = artifact
                .path
                .clone()
                .unwrap_or_else(|| initial_path.clone());
            match playback::play_audio_blocking(&target_for_play) {
                Ok(()) => played = true,
                Err(e) => playback_error = Some(e.to_string()),
            }
        }

        // Cleanup the temp file *after* playback. Always — even when
        // playback failed — so we don't leave junk in /tmp.
        if is_temp {
            if let Some(p) = artifact.path.as_ref() {
                let _ = fs::remove_file(p);
            } else {
                let _ = fs::remove_file(&initial_path);
            }
        }

        // Persist (key, model) into the audio-only slot so the next audio
        // session defaults to it. Stored separately from `last_selection`
        // and `last_image_selection`.
        let _ = self
            .session_store
            .set_last_audio_selection(&key, Some(&model))
            .await;

        if args.json {
            print_json(&artifact, &key, &model, &request, elapsed, played, is_temp);
        } else {
            print_human(
                &artifact,
                &key,
                &model,
                request.voice.as_deref(),
                played,
                is_temp,
                playback_error.as_deref(),
            );
        }
        ExitCode::Success
    }
}

async fn resolve_audio_model(
    session_store: &SessionStore,
    cache: &ModelsCache,
    args: &AudioArgs,
    key: &ApiKey,
) -> anyhow::Result<Option<String>> {
    match &args.model {
        Some(m) if !m.is_empty() => {
            let resolved = session_store.resolve_alias(m).await.unwrap_or(m.clone());
            Ok(Some(resolved))
        }
        Some(_) => pick_audio_model_interactively(cache, key, args.refresh).await,
        None => {
            if let Ok(Some(sel)) = session_store.get_last_audio_selection().await
                && sel.key_id == key.id
                && let Some(model) = sel.model
                && !model.is_empty()
            {
                return Ok(Some(model));
            }
            pick_audio_model_interactively(cache, key, args.refresh).await
        }
    }
}

/// Opens a model picker over the provider's full model list. Same approach
/// as the image picker: don't filter heuristically, let the provider error
/// be the signal — TTS model naming varies wildly across providers.
async fn pick_audio_model_interactively(
    cache: &ModelsCache,
    key: &ApiKey,
    refresh: bool,
) -> anyhow::Result<Option<String>> {
    if !std::io::stderr().is_terminal() {
        anyhow::bail!(
            "no audio model specified and no terminal available; pass -m <name> (e.g. tts-1)"
        );
    }

    let client = router_http_client();
    let all_models = crate::commands::models::fetch_all_models_cached(&client, key, cache, refresh)
        .await
        .unwrap_or_default();

    if all_models.is_empty() {
        anyhow::bail!(
            "could not fetch a model list for this key; pass -m <name> explicitly (e.g. tts-1, tts-1-hd)"
        );
    }

    Ok(crate::commands::models::prompt_model_picker(
        all_models,
        None,
        Vec::new(),
        "Select model",
    ))
}

/// Maps the user-requested format flag to a default file extension. When
/// `--format` is missing, MP3 is used (matches OpenAI's server-side default).
fn default_extension(format: Option<&str>) -> String {
    match format.map(str::to_ascii_lowercase).as_deref() {
        Some("wav") => "wav".into(),
        Some("opus") => "opus".into(),
        Some("aac") => "aac".into(),
        Some("flac") => "flac".into(),
        Some("pcm") => "pcm".into(),
        // mp3, anything unknown, or missing → .mp3 default
        _ => "mp3".into(),
    }
}

fn start_spinner_if_tty(model: &str, play_only: bool) -> Option<(Arc<AtomicBool>, JoinHandle<()>)> {
    if std::io::stderr().is_terminal() {
        let label = if play_only {
            format!(" Speaking with {}…", model)
        } else {
            format!(" Generating audio with {}…", model)
        };
        Some(style::start_spinner(Some(&label)))
    } else {
        None
    }
}

fn stop_spinner(spinner: Option<(Arc<AtomicBool>, JoinHandle<()>)>) {
    if let Some((flag, _handle)) = spinner {
        style::stop_spinner(&flag);
    }
}

fn print_human(
    artifact: &AudioArtifact,
    key: &ApiKey,
    model: &str,
    voice: Option<&str>,
    played: bool,
    is_temp: bool,
    playback_error: Option<&str>,
) {
    // Three end-states: played-only (temp), saved-only, saved-and-played.
    if is_temp {
        // Temp file is gone by now; just confirm we spoke.
        println!(
            "{} spoken ({}) via {}/{}",
            style::success_symbol(),
            voice.unwrap_or("default voice"),
            style::dim(key.display_name()),
            style::dim(model),
        );
    } else if let Some(path) = &artifact.path {
        let suffix = if played {
            format!(" ({})", style::dim("played"))
        } else if let Some(err) = playback_error {
            format!(" ({}: {})", style::yellow("playback skipped"), err)
        } else {
            String::new()
        };
        println!(
            "{} saved {} ({}, {}) via {}/{}{}",
            style::success_symbol(),
            style::cyan(path.display().to_string()),
            voice.unwrap_or("default voice"),
            human_bytes(artifact.bytes),
            style::dim(key.display_name()),
            style::dim(model),
            suffix,
        );
    }
}

fn print_json(
    artifact: &AudioArtifact,
    key: &ApiKey,
    model: &str,
    request: &AudioRequest,
    elapsed: std::time::Duration,
    played: bool,
    is_temp: bool,
) {
    let out = json!({
        "model": model,
        "key": key.display_name(),
        "voice": request.voice,
        "format": request.format,
        "speed": request.speed,
        "duration_ms": elapsed.as_millis() as u64,
        // is_temp means we discarded the file after playback; surface null
        // rather than a stale temp path that no longer exists.
        "path": if is_temp {
            None
        } else {
            artifact.path.as_ref().map(|p| p.display().to_string())
        },
        "bytes": artifact.bytes,
        "played": played,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_extension_picks_mp3_for_default_or_unknown() {
        assert_eq!(default_extension(None), "mp3");
        assert_eq!(default_extension(Some("mp3")), "mp3");
        assert_eq!(default_extension(Some("garbage")), "mp3");
    }

    #[test]
    fn default_extension_picks_explicit_formats() {
        assert_eq!(default_extension(Some("wav")), "wav");
        assert_eq!(default_extension(Some("WAV")), "wav");
        assert_eq!(default_extension(Some("opus")), "opus");
        assert_eq!(default_extension(Some("aac")), "aac");
        assert_eq!(default_extension(Some("flac")), "flac");
        assert_eq!(default_extension(Some("pcm")), "pcm");
    }
}
