use aivo::services::session_store::{decrypt, encrypt, is_encrypted};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;

#[test]
fn test_encryption_format() {
    let plaintext = "test-api-key-12345";
    let encrypted = encrypt(plaintext).unwrap();

    // Should start with enc:
    assert!(encrypted.starts_with("enc:"));

    // Should be base64 after marker
    let data = &encrypted[4..];
    let decoded = STANDARD.decode(data).unwrap();

    // Format: 16 byte IV + 16 byte auth tag + ciphertext
    assert!(decoded.len() > 32);
}

#[test]
fn test_encryption_roundtrip() {
    let test_cases = [
        "simple-key",
        "key-with-special-chars-!@#$%",
        "sk-ant-api03-test123",
        "unicode-キー-测试",
    ];

    for plaintext in test_cases {
        let encrypted = encrypt(plaintext).unwrap();
        let decrypted = decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }
}

#[test]
fn test_is_encrypted_detection() {
    assert!(is_encrypted("enc:abc123"));
    assert!(!is_encrypted("plain-text"));
    assert!(!is_encrypted(""));
    assert!(!is_encrypted("enc"));
}
