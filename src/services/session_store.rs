use aes::Aes256;
use aes_gcm::{
    aead::{generic_array::GenericArray, Aead, KeyInit},
    AesGcm,
};
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
/**
 * SessionStore service for managing credential persistence.
 * Stores credentials in ~/.config/aivo/config.json with AES-256-GCM encryption.
 */
use std::path::PathBuf;
use typenum::U16;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::errors::{CLIError, ErrorCategory};

/// Marker to identify encrypted values
pub const ENCRYPTION_MARKER: &str = "enc:";

/// Serde module for serializing/deserializing Zeroizing<String> as regular String
mod zeroizing_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use zeroize::Zeroizing;

    pub fn serialize<S>(value: &Zeroizing<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value.as_str())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Zeroizing<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Zeroizing::new(s))
    }
}

const IV_LENGTH: usize = 16;
const SALT_LENGTH: usize = 32;
const KEY_LENGTH: usize = 32;

/// Wrapper for encryption keys that automatically zeroizes on drop
#[derive(Zeroize, ZeroizeOnDrop)]
struct SecretKey([u8; KEY_LENGTH]);

impl SecretKey {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

// Use much lower iterations in tests for speed (PBKDF2 is still secure with lower iterations for testing)
#[cfg(any(test, feature = "test-fast-crypto"))]
const ITERATIONS: u32 = 100;
#[cfg(not(any(test, feature = "test-fast-crypto")))]
const ITERATIONS: u32 = 100_000;

/// API key stored on user's machine
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(with = "zeroizing_string")]
    pub key: Zeroizing<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

impl ApiKey {
    pub fn new(id: String, name: String, base_url: String, key: String) -> Self {
        Self {
            id,
            name,
            base_url,
            key: Zeroizing::new(key),
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

/// Stored configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredConfig {
    #[serde(rename = "api_keys", default)]
    pub api_keys: Vec<ApiKey>,
    #[serde(rename = "active_key_id")]
    pub active_key_id: Option<String>,
    #[serde(
        rename = "chat_model",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_model: Option<String>,
}

impl Default for StoredConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StoredConfig {
    pub fn new() -> Self {
        Self {
            api_keys: Vec::new(),
            active_key_id: None,
            chat_model: None,
        }
    }
}

/// Derives an encryption key from machine-specific information.
/// Uses username and home directory to create a consistent key per machine.
fn derive_key() -> SecretKey {
    let username = whoami::username();
    let homedir: String = dirs::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let machine_data = format!("{}:{}", username, homedir);

    // Create a static salt derived from the machine data
    let mut hasher = Sha256::new();
    hasher.update(b"aivo-salt");
    hasher.update(machine_data.as_bytes());
    let salt_full = hasher.finalize();
    let salt = &salt_full[..SALT_LENGTH];

    let iterations = ITERATIONS;
    let mut key = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(machine_data.as_bytes(), salt, iterations, &mut key);

    SecretKey(key)
}

/// Checks if a string is encrypted
pub fn is_encrypted(value: &str) -> bool {
    value.starts_with(ENCRYPTION_MARKER)
}

type Aes256Gcm16 = AesGcm<Aes256, U16, U16>;

/// Encrypts a plaintext string
pub fn encrypt(plaintext: &str) -> Result<String> {
    if plaintext.is_empty() {
        return Ok(plaintext.to_string());
    }

    // Don't double-encrypt
    if is_encrypted(plaintext) {
        return Ok(plaintext.to_string());
    }

    let key = derive_key();
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    let mut iv = [0u8; IV_LENGTH];
    rand::thread_rng().fill_bytes(&mut iv);

    let nonce = GenericArray::from_slice(&iv);

    // Encrypt
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    // Combine: IV + ciphertext (which includes auth tag in aes-gcm)
    let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
    combined.extend_from_slice(&iv);
    combined.extend_from_slice(&ciphertext);

    // Encode as base64 with marker
    Ok(format!("{}{}", ENCRYPTION_MARKER, BASE64.encode(&combined)))
}

/// Decrypts an encrypted string
pub fn decrypt(encrypted_data: &str) -> Result<String> {
    if encrypted_data.is_empty() {
        return Ok(encrypted_data.to_string());
    }

    if !is_encrypted(encrypted_data) {
        return Err(anyhow::anyhow!("Invalid encrypted data: missing marker"));
    }

    let key = derive_key();
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    // Decode from base64
    let data = BASE64
        .decode(&encrypted_data[ENCRYPTION_MARKER.len()..])
        .map_err(|e| anyhow::anyhow!("Base64 decode failed: {}", e))?;

    if data.len() < IV_LENGTH {
        return Err(anyhow::anyhow!("Invalid encrypted data: too short"));
    }

    // Extract IV and ciphertext
    let iv = &data[..IV_LENGTH];
    let ciphertext = &data[IV_LENGTH..];

    let nonce = GenericArray::from_slice(iv);

    // Decrypt
    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("Decryption failed - key may be from different machine"))?;

    String::from_utf8(plaintext)
        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in decrypted data: {}", e))
}

/// SessionStore manages API key persistence in ~/.config/aivo/config.json
#[derive(Debug, Clone)]
pub struct SessionStore {
    config_path: PathBuf,
    config_dir: PathBuf,
}

impl SessionStore {
    pub fn new() -> Self {
        let config_dir = dirs::home_dir()
            .map(|p| p.join(".config").join("aivo"))
            .unwrap_or_else(|| PathBuf::from(".config/aivo"));
        let config_path = config_dir.join("config.json");

        Self {
            config_path,
            config_dir,
        }
    }

    /// Creates a new SessionStore with a custom config path (for testing)
    #[allow(dead_code)]
    pub fn with_path(config_path: PathBuf) -> Self {
        let config_dir = config_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        Self {
            config_path,
            config_dir,
        }
    }

    /// Saves config to the config file
    pub async fn save(&self, config: &StoredConfig) -> Result<()> {
        tokio::fs::create_dir_all(&self.config_dir)
            .await
            .with_context(|| format!("Failed to create config directory: {:?}", self.config_dir))?;

        let encrypted = self.encrypt_keys(config)?;
        let data =
            serde_json::to_string_pretty(&encrypted).context("Failed to serialize config")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::write(&self.config_path, data)
                .await
                .with_context(|| format!("Failed to write config file: {:?}", self.config_path))?;
            let metadata = tokio::fs::metadata(&self.config_path).await?;
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o600);
            tokio::fs::set_permissions(&self.config_path, permissions).await?;
        }

        #[cfg(not(unix))]
        {
            tokio::fs::write(&self.config_path, data)
                .await
                .with_context(|| format!("Failed to write config file: {:?}", self.config_path))?;
        }

        Ok(())
    }

    /// Loads config from the config file
    pub async fn load(&self) -> Result<StoredConfig> {
        let data = match tokio::fs::read_to_string(&self.config_path).await {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(StoredConfig::new());
            }
            Err(e) => return Err(e.into()),
        };

        let parsed: StoredConfig = match serde_json::from_str(&data) {
            Ok(p) => p,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "config file is corrupted and cannot be read: {e}"
                ))
            }
        };

        self.decrypt_keys(&parsed)
    }

    /// Adds a new API key and returns its generated ID
    pub async fn add_key(&self, name: &str, base_url: &str, key: &str) -> Result<String> {
        let mut config = self.load().await?;

        let existing_ids: HashSet<String> = config.api_keys.iter().map(|k| k.id.clone()).collect();
        let id = loop {
            let id = format!("{:04x}", rand::random::<u16>());
            if !existing_ids.contains(&id) {
                break id;
            }
        };

        config.api_keys.push(ApiKey::new(
            id.clone(),
            name.to_string(),
            base_url.to_string(),
            key.to_string(),
        ));

        self.save(&config).await?;
        Ok(id)
    }

    /// Gets all API keys
    pub async fn get_keys(&self) -> Result<Vec<ApiKey>> {
        Ok(self.load().await?.api_keys)
    }

    /// Gets a specific API key by ID
    #[allow(dead_code)]
    pub async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        let keys = self.get_keys().await?;
        Ok(keys.into_iter().find(|k| k.id == id))
    }

    /// Deletes an API key by ID
    pub async fn delete_key(&self, id: &str) -> Result<bool> {
        let mut config = self.load().await?;
        let initial_len = config.api_keys.len();
        config.api_keys.retain(|k| k.id != id);

        if config.api_keys.len() < initial_len {
            if config.active_key_id.as_deref() == Some(id) {
                config.active_key_id = None;
            }
            self.save(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Updates an existing API key's fields by ID. Returns false if not found.
    pub async fn update_key(
        &self,
        id: &str,
        name: &str,
        base_url: &str,
        key: &str,
    ) -> Result<bool> {
        let mut config = self.load().await?;
        if let Some(entry) = config.api_keys.iter_mut().find(|k| k.id == id) {
            entry.name = name.to_string();
            entry.base_url = base_url.to_string();
            entry.key = Zeroizing::new(key.to_string());
            self.save(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Sets the currently active API key
    pub async fn set_active_key(&self, id: &str) -> Result<()> {
        let mut config = self.load().await?;

        if !config.api_keys.iter().any(|k| k.id == id) {
            return Err(CLIError::new(
                format!("Key {} not found", id),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys list' to see available keys"),
            )
            .into());
        }

        config.active_key_id = Some(id.to_string());
        self.save(&config).await
    }

    /// Resolves an API key by ID or name.
    /// Tries exact ID match first, then name match.
    /// Returns an error if no match found or multiple names match.
    pub async fn resolve_key_by_id_or_name(&self, id_or_name: &str) -> Result<ApiKey> {
        let keys = self.get_keys().await?;

        // Try exact ID match first
        if let Some(key) = keys.iter().find(|k| k.id == id_or_name) {
            return Ok(key.clone());
        }

        // Try name match
        let name_matches: Vec<_> = keys.iter().filter(|k| k.name == id_or_name).collect();

        match name_matches.len() {
            0 => Err(CLIError::new(
                format!("API key \"{}\" not found", id_or_name),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys list' to see available keys"),
            )
            .into()),
            1 => Ok(name_matches[0].clone()),
            _ => Err(CLIError::new(
                format!(
                    "Multiple keys found with name \"{}\". Use the key ID instead.",
                    id_or_name
                ),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys list' to see key IDs"),
            )
            .into()),
        }
    }

    /// Gets the currently active API key
    pub async fn get_active_key(&self) -> Result<Option<ApiKey>> {
        let config = self.load().await?;

        match config.active_key_id {
            Some(ref id) => Ok(config.api_keys.into_iter().find(|k| k.id == *id)),
            None => Ok(None),
        }
    }

    /// Gets the config path
    #[allow(dead_code)]
    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    /// Gets the persisted chat model
    pub async fn get_chat_model(&self) -> Result<Option<String>> {
        let config = self.load().await?;
        Ok(config.chat_model)
    }

    /// Saves the chat model to config
    pub async fn set_chat_model(&self, model: &str) -> Result<()> {
        let mut config = self.load().await?;
        config.chat_model = Some(model.to_string());
        self.save(&config).await
    }

    /// Encrypts API keys before saving
    fn encrypt_keys(&self, config: &StoredConfig) -> Result<StoredConfig> {
        let mut encrypted = config.clone();
        for key in &mut encrypted.api_keys {
            if !is_encrypted(&key.key) {
                key.key = Zeroizing::new(encrypt(&key.key)?);
            }
        }
        Ok(encrypted)
    }

    /// Decrypts API keys after loading
    fn decrypt_keys(&self, config: &StoredConfig) -> Result<StoredConfig> {
        let mut decrypted = config.clone();
        for key in &mut decrypted.api_keys {
            if is_encrypted(&key.key) {
                let plaintext = decrypt(&key.key)
                    .with_context(|| format!("failed to decrypt key '{}'", key.name))?;
                key.key = Zeroizing::new(plaintext);
            }
        }
        Ok(decrypted)
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_load_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let config = store.load().await.unwrap();
        assert!(config.api_keys.is_empty());
        assert!(config.active_key_id.is_none());
    }

    #[tokio::test]
    async fn test_key_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // Add a key
        let id = store
            .add_key("my-key", "http://localhost:8080", "sk-test123")
            .await
            .unwrap();
        assert_eq!(id.len(), 4);

        // Verify it was saved
        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "my-key");
        assert_eq!(keys[0].base_url, "http://localhost:8080");

        // Set as active
        store.set_active_key(&id).await.unwrap();
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().id, id);

        // Delete the key
        assert!(store.delete_key(&id).await.unwrap());
        let keys = store.get_keys().await.unwrap();
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn test_key_encryption_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        let id = store
            .add_key("test", "http://localhost", "sk-secret-12345")
            .await
            .unwrap();

        // Verify the file contains encrypted key
        let file_content = tokio::fs::read_to_string(&config_path).await.unwrap();
        assert!(file_content.contains("enc:"));
        assert!(!file_content.contains("sk-secret-12345"));

        // Verify we can still read back the decrypted key
        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.key.as_str(), "sk-secret-12345");
    }

    #[tokio::test]
    async fn test_delete_active_key_clears_selection() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key("my-key", "http://localhost:8080", "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        // Delete the active key
        store.delete_key(&id).await.unwrap();

        // Active key should be cleared
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn test_resolve_key_by_id() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key("my-key", "http://localhost:8080", "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name(&id).await.unwrap();
        assert_eq!(resolved.id, id);
        assert_eq!(resolved.name, "my-key");
    }

    #[tokio::test]
    async fn test_resolve_key_by_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key("my-key", "http://localhost:8080", "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name("my-key").await.unwrap();
        assert_eq!(resolved.id, id);
    }

    #[tokio::test]
    async fn test_resolve_key_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let result = store.resolve_key_by_id_or_name("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_resolve_key_ambiguous_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        store
            .add_key("same-name", "http://localhost:8080", "sk-test1")
            .await
            .unwrap();
        store
            .add_key("same-name", "http://localhost:9090", "sk-test2")
            .await
            .unwrap();

        let result = store.resolve_key_by_id_or_name("same-name").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple keys found"));
    }

    #[tokio::test]
    async fn test_load_corrupted_config_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(&config_path, b"not valid json {{{")
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        let result = store.load().await;
        assert!(result.is_err(), "expected Err on corrupted config, got Ok");
    }

    #[tokio::test]
    async fn test_load_returns_error_on_invalid_encrypted_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        // "enc:" prefix triggers decryption; the payload is not valid ciphertext
        let bad_config = r#"{"api_keys":[{"id":"aaaa","name":"test","baseUrl":"http://example.com","key":"enc:notvalidbase64!!!","createdAt":"2024-01-01T00:00:00Z"}],"active_key_id":null}"#;
        tokio::fs::write(&config_path, bad_config.as_bytes())
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        let result = store.load().await;
        assert!(
            result.is_err(),
            "expected Err on invalid encrypted key, got Ok"
        );
    }

    #[tokio::test]
    async fn test_update_key_fields() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key("original", "http://localhost:8080", "sk-old")
            .await
            .unwrap();

        let updated = store
            .update_key(&id, "renamed", "https://new.example.com", "sk-new")
            .await
            .unwrap();
        assert!(updated);

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.name, "renamed");
        assert_eq!(key.base_url, "https://new.example.com");
        assert_eq!(key.key.as_str(), "sk-new");
        assert_eq!(key.id, id);
    }

    #[tokio::test]
    async fn test_update_key_not_found_returns_false() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let updated = store
            .update_key("nonexistent", "name", "http://example.com", "sk-key")
            .await
            .unwrap();
        assert!(!updated);
    }

    #[tokio::test]
    async fn test_update_key_preserves_created_at() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key("orig", "http://localhost", "sk-test")
            .await
            .unwrap();
        let before = store.get_key_by_id(&id).await.unwrap().unwrap();

        store
            .update_key(&id, "new-name", "http://localhost", "sk-test")
            .await
            .unwrap();
        let after = store.get_key_by_id(&id).await.unwrap().unwrap();

        assert_eq!(before.created_at, after.created_at);
    }
}
