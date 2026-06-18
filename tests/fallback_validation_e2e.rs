/**
 * E2E validation tests for the fallback CLI.
 *
 * Tests that the CLI enforces fallback naming and content constraints:
 *   1. Empty name is rejected
 *   2. Whitespace-only name is rejected
 *   3. '@' in name is rejected (ambiguity with @fallback reference syntax)
 *   4. ':' in name is rejected (conflict with provider:model format)
 *   5. ':' in @fallback reference target is rejected
 *   6. Empty provider in target is rejected
 *   7. Valid fallback creation succeeds
 *   8. Valid fallback removal works
 *
 * These tests spawn the compiled aivo binary and operate on an isolated
 * config directory (via HOME override), so they don't touch real config.
 */

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Returns an `aivo` command configured with a temporary HOME for isolation.
/// The caller keeps `_guard` alive to prevent the temp dir from being deleted
/// before the command runs.
fn aivo_cmd() -> (Command, TempDir) {
    let temp = TempDir::new().expect("failed to create temp dir");
    let mut cmd = Command::cargo_bin("aivo").expect("aivo binary not found — build first");
    cmd.env("HOME", temp.path());
    (cmd, temp)
}

/// Shorthand for a fallback set command with isolated home.
fn aivo_fallback_set(fallback_name: &str, targets: &[&str]) -> (Command, TempDir) {
    let (mut cmd, guard) = aivo_cmd();
    cmd.arg("fallback").arg("--set").arg(fallback_name).arg("--");
    for t in targets {
        cmd.arg(t);
    }
    (cmd, guard)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn reject_empty_fallback_name() {
    let (mut cmd, _guard) = aivo_fallback_set("", &["anthropic:claude-sonnet-4-6"]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("cannot be empty"));
}

#[test]
fn reject_whitespace_fallback_name() {
    let (mut cmd, _guard) = aivo_fallback_set("   ", &["anthropic:claude-sonnet-4-6"]);
    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("cannot be empty")
                .or(predicate::str::contains("whitespace")),
        );
}

#[test]
fn reject_at_sign_in_fallback_name() {
    let (mut cmd, _guard) = aivo_fallback_set("@primary", &["anthropic:claude-sonnet-4-6"]);
    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("reserved")
                .or(predicate::str::contains("'@'"))
                .or(predicate::str::contains("@")),
        );
}

#[test]
fn reject_colon_in_fallback_name() {
    let (mut cmd, _guard) = aivo_fallback_set("bad:name", &["anthropic:claude-sonnet-4-6"]);
    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("reserved")
                .or(predicate::str::contains("colon"))
                .or(predicate::str::contains("':'")),
        );
}

#[test]
fn reject_colon_in_fallback_reference_target() {
    let (mut cmd, _guard) = aivo_fallback_set("test-fb", &["@bad:ref"]);
    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("reserved")
                .or(predicate::str::contains("colon"))
                .or(predicate::str::contains("':'")),
        );
}

#[test]
fn reject_empty_provider_in_target() {
    let (mut cmd, _guard) = aivo_fallback_set("test-fb", &[":gpt-4o"]);
    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("empty provider")
                .or(predicate::str::contains("Invalid target")),
        );
}

#[test]
fn valid_fallback_creation_appears_in_list() {
    let (mut cmd, guard) = aivo_fallback_set("e2e-valid", &["anthropic:claude-sonnet-4-6"]);
    cmd.assert().success();

    // List fallbacks and verify the entry appears
    let mut list_cmd = Command::cargo_bin("aivo").expect("aivo binary not found");
    list_cmd.env("HOME", guard.path()).arg("fallback");
    list_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("e2e-valid"));
}

#[test]
fn valid_fallback_removed_disappears_from_list() {
    let guard = TempDir::new().expect("failed to create temp dir");

    // Create
    let mut create = Command::cargo_bin("aivo").expect("aivo binary not found");
    create
        .env("HOME", guard.path())
        .arg("fallback")
        .arg("--set")
        .arg("e2e-valid")
        .arg("--")
        .arg("anthropic:claude-sonnet-4-6");
    create.assert().success();

    // Remove
    let mut remove = Command::cargo_bin("aivo").expect("aivo binary not found");
    remove
        .env("HOME", guard.path())
        .arg("fallback")
        .arg("--rm")
        .arg("e2e-valid");
    remove.assert().success();

    // Verify it's gone
    let mut list = Command::cargo_bin("aivo").expect("aivo binary not found");
    list.env("HOME", guard.path()).arg("fallback");
    list.assert()
        .success()
        .stdout(predicate::str::contains("e2e-valid").not());
}
