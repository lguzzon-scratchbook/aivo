//! Zlib-compresses embedded text assets into OUT_DIR — keeps ~140 KB of
//! .rodata off the 10 MiB musl release gate. Runtime side: services/embedded_assets.rs.

use std::env;
use std::fs;
use std::path::Path;

const ASSETS: &[(&str, &str)] = &[
    ("src/data/model_limits.json", "model_limits.json.z"),
    ("src/data/providers.json", "providers.json.z"),
    ("src/commands/aivo_guide.md", "aivo_guide.md.z"),
    (
        "src/agent/builtin_skills/create-skill.md",
        "create-skill.md.z",
    ),
    (
        "src/agent/builtin_skills/create-agent.md",
        "create-agent.md.z",
    ),
    (
        "src/agent/builtin_agents/explorer.md",
        "agent-explorer.md.z",
    ),
    (
        "src/agent/builtin_agents/aivo-guide.md",
        "agent-aivo-guide.md.z",
    ),
    (
        "src/agent/builtin_agents/verification.md",
        "agent-verification.md.z",
    ),
    ("src/agent/builtin_agents/advisor.md", "agent-advisor.md.z"),
    (
        "src/agent/builtin_agents/evaluate.md",
        "agent-evaluate.md.z",
    ),
];

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    for (src, out) in ASSETS {
        println!("cargo:rerun-if-changed={src}");
        let data = fs::read(src).unwrap_or_else(|e| panic!("read {src}: {e}"));
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&data, 10);
        fs::write(Path::new(&out_dir).join(out), packed)
            .unwrap_or_else(|e| panic!("write {out}: {e}"));
    }
}
