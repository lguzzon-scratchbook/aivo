use super::*;

#[cfg(target_os = "macos")]
use std::process::Command;

pub(super) fn read_system_clipboard() -> Result<ClipboardPayload> {
    #[cfg(target_os = "macos")]
    {
        if let Some(attachment) = read_macos_clipboard_image()? {
            return Ok(ClipboardPayload::Attachment(attachment));
        }

        let text = read_command_stdout("pbpaste", &[])?;
        if text.is_empty() {
            Ok(ClipboardPayload::Empty)
        } else {
            Ok(ClipboardPayload::Text(text))
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(ClipboardPayload::Empty)
    }
}

#[cfg(target_os = "macos")]
pub(super) fn read_command_stdout(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| anyhow::anyhow!("Failed to run '{}': {err}", program))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            anyhow::bail!("'{}' exited with {}", program, output.status);
        }
        anyhow::bail!("'{}' failed: {}", program, stderr);
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(target_os = "macos")]
pub(super) fn read_macos_clipboard_image() -> Result<Option<MessageAttachment>> {
    let script = r#"import AppKit
import Foundation

let pasteboard = NSPasteboard.general
if let data = pasteboard.data(forType: .png) {
    print(data.base64EncodedString())
} else if
    let tiff = pasteboard.data(forType: .tiff),
    let image = NSImage(data: tiff),
    let tiffData = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiffData),
    let png = bitmap.representation(using: .png, properties: [:])
{
    print(png.base64EncodedString())
}
"#;

    let mut command = Command::new("swift");
    command.env("CLANG_MODULE_CACHE_PATH", "/tmp/clang-module-cache");
    command.arg("-e").arg(script);

    let output = command
        .output()
        .map_err(|err| anyhow::anyhow!("Failed to access clipboard image: {err}"))?;
    if !output.status.success() {
        return Ok(None);
    }

    let data = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if data.is_empty() {
        return Ok(None);
    }

    Ok(Some(MessageAttachment {
        name: format!("clipboard-{}.png", Utc::now().format("%Y%m%d-%H%M%S")),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline { data },
    }))
}

pub(super) fn parse_slash_command(input: &str) -> Result<SlashCommand> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    match command {
        "new" => Ok(SlashCommand::New),
        "exit" => Ok(SlashCommand::Exit),
        "resume" => Ok(SlashCommand::Resume(argument)),
        "model" => Ok(SlashCommand::Model(argument)),
        "key" => Ok(SlashCommand::Key(argument)),
        "attach" => Ok(SlashCommand::Attach(
            argument.ok_or_else(|| anyhow::anyhow!("Usage: /attach <path>"))?,
        )),
        "detach" => Ok(SlashCommand::Detach(
            argument
                .ok_or_else(|| anyhow::anyhow!("Usage: /detach <n>"))?
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("Usage: /detach <n>"))?,
        )),
        "help" => Ok(SlashCommand::Help),
        "" => anyhow::bail!("Type a command after '/'"),
        other => anyhow::bail!("Unknown command '/{other}'"),
    }
}

pub(super) fn reduce_motion_requested() -> bool {
    env::var("AIVO_REDUCE_MOTION")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn is_help_shortcut(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::F(1))
}

pub(super) fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

pub(super) fn copilot_token_manager_for_key(key: &ApiKey) -> Option<Arc<CopilotTokenManager>> {
    if key.base_url == "copilot" {
        Some(Arc::new(CopilotTokenManager::new(
            key.key.as_str().to_string(),
        )))
    } else {
        None
    }
}
