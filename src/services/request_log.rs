//! Request logger for `aivo serve` — writes JSONL entries to ~/.config/aivo/logs/.
//!
//! Each entry contains: timestamp, path, model, status, latency_ms.
//! Response bodies are never logged (privacy). Logging failures are non-fatal.

use chrono::Utc;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Debug, Serialize)]
pub struct RequestLogEntry {
    pub timestamp: String,
    pub path: String,
    pub model: Option<String>,
    pub status: u16,
    pub latency_ms: u64,
}

/// Async JSONL request logger. Writes are buffered behind a mutex.
/// All errors are silently ignored — logging must never crash the server.
#[derive(Clone)]
pub struct RequestLogger {
    inner: Arc<Mutex<LogWriter>>,
}

struct LogWriter {
    file: Option<tokio::fs::File>,
    warned: bool,
}

impl RequestLogger {
    /// Creates a new logger that writes to ~/.config/aivo/logs/serve-YYYY-MM-DD.jsonl.
    /// Returns None if the log directory can't be created (non-fatal).
    pub async fn new(config_dir: &std::path::Path) -> Option<Self> {
        let log_dir = config_dir.join("logs");
        if let Err(e) = tokio::fs::create_dir_all(&log_dir).await {
            eprintln!("  Warning: could not create log directory: {}", e);
            return None;
        }

        let log_path = log_file_path(&log_dir);
        let file = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Warning: could not open log file: {}", e);
                return None;
            }
        };

        eprintln!("  Logging to {}", log_path.display());

        Some(Self {
            inner: Arc::new(Mutex::new(LogWriter {
                file: Some(file),
                warned: false,
            })),
        })
    }

    /// Logs a request entry. Failures are silently ignored.
    pub async fn log(&self, entry: RequestLogEntry) {
        let mut writer = self.inner.lock().await;
        if writer.warned || writer.file.is_none() {
            return;
        }

        let mut line = match serde_json::to_string(&entry) {
            Ok(s) => s,
            Err(_) => return,
        };
        line.push('\n');

        if let Err(e) = writer
            .file
            .as_mut()
            .unwrap()
            .write_all(line.as_bytes())
            .await
        {
            eprintln!("  Warning: log write failed: {}", e);
            writer.warned = true;
        }
    }
}

fn log_file_path(log_dir: &std::path::Path) -> PathBuf {
    let date = Utc::now().format("%Y-%m-%d");
    log_dir.join(format!("serve-{}.jsonl", date))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_entry_serializes_to_json() {
        let entry = RequestLogEntry {
            timestamp: "2026-03-20T12:00:00Z".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: Some("claude-sonnet-4-6".to_string()),
            status: 200,
            latency_ms: 1234,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"path\":\"/v1/chat/completions\""));
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"latency_ms\":1234"));
    }

    #[test]
    fn log_entry_with_no_model() {
        let entry = RequestLogEntry {
            timestamp: "2026-03-20T12:00:00Z".to_string(),
            path: "/health".to_string(),
            model: None,
            status: 200,
            latency_ms: 1,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"model\":null"));
    }

    #[test]
    fn log_file_path_uses_date() {
        let dir = std::path::Path::new("/tmp/logs");
        let path = log_file_path(dir);
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("serve-"));
        assert!(name.ends_with(".jsonl"));
    }

    #[tokio::test]
    async fn logger_writes_to_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let logger = RequestLogger::new(dir.path()).await.unwrap();

        logger
            .log(RequestLogEntry {
                timestamp: "2026-03-20T12:00:00Z".to_string(),
                path: "/health".to_string(),
                model: None,
                status: 200,
                latency_ms: 1,
            })
            .await;

        // Force flush by dropping
        drop(logger);

        // Read any jsonl file in the logs dir
        let log_dir = dir.path().join("logs");
        let mut entries = tokio::fs::read_dir(&log_dir).await.unwrap();
        let entry = entries.next_entry().await.unwrap().unwrap();
        let content = tokio::fs::read_to_string(entry.path()).await.unwrap();
        assert!(content.contains("/health"));
    }
}
