//! Accessors for the zlib-compressed assets `build.rs` emits into OUT_DIR.

use std::sync::OnceLock;

fn inflate(packed: &[u8]) -> String {
    let bytes = zune_inflate::DeflateDecoder::new(packed)
        .decode_zlib()
        .expect("embedded asset: corrupt zlib stream");
    String::from_utf8(bytes).expect("embedded asset: not UTF-8")
}

macro_rules! embedded_asset {
    ($fn_name:ident, $file:literal) => {
        pub fn $fn_name() -> &'static str {
            static CELL: OnceLock<String> = OnceLock::new();
            CELL.get_or_init(|| inflate(include_bytes!(concat!(env!("OUT_DIR"), "/", $file))))
                .as_str()
        }
    };
}

embedded_asset!(model_limits_json, "model_limits.json.z");
embedded_asset!(providers_json, "providers.json.z");
embedded_asset!(aivo_guide_md, "aivo_guide.md.z");
embedded_asset!(create_skill_md, "create-skill.md.z");
embedded_asset!(create_agent_md, "create-agent.md.z");
embedded_asset!(agent_explorer_md, "agent-explorer.md.z");
embedded_asset!(agent_aivo_guide_md, "agent-aivo-guide.md.z");
embedded_asset!(agent_verification_md, "agent-verification.md.z");
embedded_asset!(agent_advisor_md, "agent-advisor.md.z");
embedded_asset!(agent_evaluate_md, "agent-evaluate.md.z");
