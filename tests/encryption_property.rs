//! Property-based tests for encryption

use aivo::services::session_store::{decrypt, encrypt, is_encrypted};

#[test]
fn test_encryption_never_panics() {
    // Test various input types - keep tests fast with moderate input sizes
    let inputs = [
        "a",
        "normal-key",
        "key-with-symbols!@#",
        "sk-test123456789",
        "unicode-キー-测试",
    ];

    for input in inputs {
        let encrypted = encrypt(input).expect("encryption should not fail");
        assert!(is_encrypted(&encrypted));

        let decrypted = decrypt(&encrypted).expect("decryption should not fail");
        assert_eq!(decrypted, input);
    }

    // Empty string special case - returns empty without encryption
    assert_eq!(encrypt("").unwrap(), "");
}

#[test]
fn test_double_encryption_idempotent() {
    let key = "my-api-key";
    let encrypted1 = encrypt(key).unwrap();
    let encrypted2 = encrypt(&encrypted1).unwrap();

    // Double encryption should return the same value
    assert_eq!(encrypted1, encrypted2);
}
