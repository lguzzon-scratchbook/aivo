use std::process::Command;
use std::path::PathBuf;
use tempfile::TempDir;

fn get_binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_aivo"))
}

#[test]
fn test_cli_help() {
    let output = Command::new(get_binary_path())
        .arg("--help")
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("aivo"));
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("Commands:"));
    assert!(stdout.contains("run"));
    assert!(stdout.contains("keys"));
}

#[test]
fn test_cli_version() {
    let output = Command::new(get_binary_path())
        .arg("--version")
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("aivo"));
}

#[test]
fn test_keys_list_without_auth() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join(".config").join("aivo");
    std::fs::create_dir_all(&config_dir).unwrap();

    // Create empty config
    let config_path = config_dir.join("config.json");
    std::fs::write(&config_path, r#"{"access_token": "", "user": {"name": "", "email": ""}, "stored_at": ""}"#).unwrap();

    let output = Command::new(get_binary_path())
        .arg("keys")
        .arg("list")
        .env("HOME", temp_dir.path())
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No API keys found") || stdout.contains("Keys:"));
}
