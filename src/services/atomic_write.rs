use anyhow::{Context, Result};
use std::path::Path;

/// Atomically writes `data` to `final_path` with mode `0o600` on Unix.
///
/// Writes to a sibling `*.tmp` file created with restrictive permissions from
/// the start (no world-readable window), fsyncs, then renames into place.
/// Callers must ensure the parent directory exists.
pub(crate) async fn atomic_write_secure(final_path: &Path, data: Vec<u8>) -> Result<()> {
    let tmp_path = final_path.with_extension("json.tmp");
    let tmp_for_write = tmp_path.clone();

    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp_for_write)?;
        file.write_all(&data)?;
        file.sync_all()?;
        Ok(())
    })
    .await
    .context("Join error writing temp file")?
    .with_context(|| format!("Failed to write temp file: {:?}", tmp_path))?;

    tokio::fs::rename(&tmp_path, final_path)
        .await
        .with_context(|| format!("Failed to rename temp file to {:?}", final_path))?;

    Ok(())
}
