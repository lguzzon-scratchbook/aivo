fn main() {
    // Simple build script - just set the build timestamp
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    // Try to get git hash if available
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if output.status.success() {
            let git_sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rustc-env=VERGEN_GIT_SHA={}", git_sha);
        }
    }

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=src/");
}
