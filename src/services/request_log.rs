//! Request logger for `aivo serve` — writes JSONL entries to ~/.config/aivo/logs/.
//!
//! Each entry contains: timestamp, path, model, status, latency_ms.
//! Response bodies are never logged (privacy). Logging failures are non-fatal.

use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize)]
pub struct RequestLogEntry {
    pub timestamp: String,
    pub method: String,
    pub path: String,
    pub model: Option<String>,
    pub status: u16,
    pub latency_ms: u64,
    pub ip: String,
}

/// Async JSONL request logger. Writes are buffered behind a mutex.
/// All errors are silently ignored — logging must never crash the server.
#[derive(Clone)]
pub struct RequestLogger {
    inner: Arc<Mutex<LogWriter>>,
    display: String,
}

enum LogTarget {
    File(tokio::fs::File),
    Stdout,
}

struct LogWriter {
    target: LogTarget,
    warned: bool,
}

impl RequestLogger {
    /// Creates a logger that writes JSONL to stdout.
    pub fn new_stdout() -> Self {
        Self {
            inner: Arc::new(Mutex::new(LogWriter {
                target: LogTarget::Stdout,
                warned: false,
            })),
            display: "stdout".to_string(),
        }
    }

    /// Creates a new logger that writes to a specific file path.
    /// Parent directories are created automatically.
    pub async fn new_with_path(path: &std::path::Path) -> Option<Self> {
        if let Some(parent) = path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            eprintln!("  Warning: could not create log directory: {}", e);
            return None;
        }

        let file = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Warning: could not open log file: {}", e);
                return None;
            }
        };

        let display = path.display().to_string();
        Some(Self {
            inner: Arc::new(Mutex::new(LogWriter {
                target: LogTarget::File(file),
                warned: false,
            })),
            display,
        })
    }

    /// Returns the log output target for display in startup output.
    pub fn path_display(&self) -> &str {
        &self.display
    }

    /// Logs a request entry. Failures are silently ignored.
    pub async fn log(&self, entry: RequestLogEntry) {
        let mut writer = self.inner.lock().await;
        if writer.warned {
            return;
        }

        let mut line = match serde_json::to_string(&entry) {
            Ok(s) => s,
            Err(_) => return,
        };
        line.push('\n');

        let result = match &mut writer.target {
            LogTarget::File(f) => {
                use tokio::io::AsyncWriteExt;
                match f.write_all(line.as_bytes()).await {
                    Ok(()) => f.flush().await,
                    Err(e) => Err(e),
                }
            }
            LogTarget::Stdout => {
                use std::io::Write;
                std::io::stdout()
                    .write_all(line.as_bytes())
                    .map_err(|e| std::io::Error::new(e.kind(), e))
            }
        };

        if let Err(e) = result {
            eprintln!("  Warning: log write failed: {}", e);
            writer.warned = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_entry_serializes_to_json() {
        let entry = RequestLogEntry {
            timestamp: "2026-03-20T12:00:00Z".to_string(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            model: Some("claude-sonnet-4-6".to_string()),
            status: 200,
            latency_ms: 1234,
            ip: "127.0.0.1".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"method\":\"POST\""));
        assert!(json.contains("\"path\":\"/v1/chat/completions\""));
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"latency_ms\":1234"));
        assert!(json.contains("\"ip\":\"127.0.0.1\""));
    }

    #[test]
    fn log_entry_with_no_model() {
        let entry = RequestLogEntry {
            timestamp: "2026-03-20T12:00:00Z".to_string(),
            method: "GET".to_string(),
            path: "/health".to_string(),
            model: None,
            status: 200,
            latency_ms: 1,
            ip: "::1".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"model\":null"));
    }

    #[test]
    fn stdout_logger_display() {
        let logger = RequestLogger::new_stdout();
        assert_eq!(logger.path_display(), "stdout");
    }

    #[tokio::test]
    async fn logger_writes_to_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        let logger = RequestLogger::new_with_path(&log_path).await.unwrap();

        logger
            .log(RequestLogEntry {
                timestamp: "2026-03-20T12:00:00Z".to_string(),
                method: "GET".to_string(),
                path: "/health".to_string(),
                model: None,
                status: 200,
                latency_ms: 1,
                ip: "127.0.0.1".to_string(),
            })
            .await;

        drop(logger);

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(content.contains("/health"));
    }
}
