//! Per-(tool, model) protocol route cache: one per running router, replacing
//! the process-wide `active_protocol` pin so a multi-model gateway key learns a
//! route per model instead of thrashing one scalar. Keyed per `(tool, key,
//! model)` — tool is implicit in the owning router. With no learned route a slot
//! starts at the router's `tool_native` protocol (claude→messages, codex→chat/
//! responses, gemini→google) and the existing cascade walks from there.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

use super::model_names::strip_context_suffix;
use super::provider_protocol::{PathVariant, ProviderProtocol, decode_route, encode_route};

/// A route learned for one model, persisted losslessly per `(tool, model)` on
/// the owning `ApiKey`. `pathVariant` is omitted when it's the default so the
/// stored JSON stays compact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedRoute {
    pub protocol: String,
    #[serde(
        rename = "pathVariant",
        default,
        skip_serializing_if = "is_default_variant"
    )]
    pub path_variant: String,
}

fn is_default_variant(v: &str) -> bool {
    v.is_empty() || v == "default"
}

impl PersistedRoute {
    pub fn from_route(protocol: ProviderProtocol, variant: PathVariant) -> Self {
        Self {
            protocol: protocol.as_str().to_string(),
            path_variant: variant.as_str().to_string(),
        }
    }

    /// Pack into the active-route byte; `None` for an unrecognised protocol so a
    /// stale entry is ignored rather than poisoning the seed.
    pub fn to_byte(&self) -> Option<u8> {
        let protocol = ProviderProtocol::parse(&self.protocol)?;
        let variant = PathVariant::parse(&self.path_variant).unwrap_or(PathVariant::Default);
        Some(encode_route(protocol, variant))
    }
}

/// The learned route for a single model within one router process.
pub struct RouteSlot {
    /// Packed `(protocol, path_variant)` the cascade reads/writes via `protocol_fallback`.
    route: AtomicU8,
    /// Set on an authoritative response (2xx or structured semantic rejection):
    /// proves the route, seeds `route_proven`, and gates persistence.
    confirmed: AtomicBool,
    consecutive_failures: AtomicU8,
    /// Reset target after a failure streak (this slot's seed).
    seed: u8,
    /// Route already persisted at seed time; `needs_persist` diffs against it.
    persisted_seed: Option<u8>,
}

impl RouteSlot {
    pub fn route_atom(&self) -> &AtomicU8 {
        &self.route
    }

    pub fn failures_atom(&self) -> &AtomicU8 {
        &self.consecutive_failures
    }

    pub fn current(&self) -> (ProviderProtocol, PathVariant) {
        decode_route(self.route.load(Ordering::Relaxed))
    }

    pub fn seed_route(&self) -> (ProviderProtocol, PathVariant) {
        decode_route(self.seed)
    }

    pub fn is_confirmed(&self) -> bool {
        self.confirmed.load(Ordering::Relaxed)
    }

    /// Mark the current route proven after an authoritative outcome.
    pub fn confirm(&self) {
        self.confirmed.store(true, Ordering::Relaxed);
    }

    /// Confirmed and new-or-changed vs what's stored — i.e. worth writing.
    fn needs_persist(&self) -> bool {
        self.confirmed.load(Ordering::Relaxed)
            && self.persisted_seed != Some(self.route.load(Ordering::Relaxed))
    }
}

/// Per-process route memory for one router. Seeded from the launch key's
/// persisted routes for this tool; the source of truth while the process runs.
pub struct RouteCache {
    tool: &'static str,
    /// Fallback when neither a per-model nor a `""` default route is seeded.
    tool_native: u8,
    /// Canonical model (`""` = tool default) → packed route, from the key.
    seed: HashMap<String, u8>,
    slots: Mutex<HashMap<String, Arc<RouteSlot>>>,
}

impl RouteCache {
    pub fn new(
        tool: &'static str,
        tool_native: ProviderProtocol,
        seed_routes: BTreeMap<String, PersistedRoute>,
    ) -> Self {
        let mut seed = HashMap::new();
        for (model, route) in seed_routes {
            if let Some(byte) = route.to_byte() {
                seed.insert(canonical_model(&model), byte);
            }
        }
        Self {
            tool,
            tool_native: encode_route(tool_native, PathVariant::Default),
            seed,
            slots: Mutex::new(HashMap::new()),
        }
    }

    pub fn tool(&self) -> &'static str {
        self.tool
    }

    /// Get (or lazily create) the slot for `raw_model`. The slot's starting
    /// route is the model's persisted route, else the `""` default, else the
    /// tool-native protocol. A slot seeded from a per-model persisted route
    /// starts `confirmed` so a transient error won't fan out across protocols.
    pub fn resolve(&self, raw_model: &str) -> Arc<RouteSlot> {
        let model = canonical_model(raw_model);
        let mut slots = self.slots.lock().unwrap();
        if let Some(slot) = slots.get(&model) {
            return slot.clone();
        }
        let persisted_seed = self.seed.get(&model).copied();
        let seed = persisted_seed
            .or_else(|| self.seed.get("").copied())
            .unwrap_or(self.tool_native);
        let slot = Arc::new(RouteSlot {
            route: AtomicU8::new(seed),
            confirmed: AtomicBool::new(persisted_seed.is_some()),
            consecutive_failures: AtomicU8::new(0),
            seed,
            persisted_seed,
        });
        slots.insert(model, slot.clone());
        slot
    }

    /// Models whose confirmed route is new or changed this session, ready to be
    /// merged back into the key. Skips unconfirmed slots (e.g. a bad-key session
    /// that only ever saw failures) so they can't poison the persisted route.
    pub fn dirty_routes(&self) -> Vec<(String, PersistedRoute)> {
        let slots = self.slots.lock().unwrap();
        slots
            .iter()
            .filter(|(_, slot)| slot.needs_persist())
            .map(|(model, slot)| {
                let (protocol, variant) = slot.current();
                (model.clone(), PersistedRoute::from_route(protocol, variant))
            })
            .collect()
    }
}

/// Collapse the variants of a model name that should share one route entry:
/// strip the `[<N>m]` context suffix Claude Code appends, then trim. Prevents
/// `claude-sonnet` and `claude-sonnet[1m]` fragmenting into two entries.
pub fn canonical_model(raw: &str) -> String {
    strip_context_suffix(raw.trim()).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(protocol: &str) -> PersistedRoute {
        PersistedRoute {
            protocol: protocol.to_string(),
            path_variant: String::new(),
        }
    }

    fn seeded(pairs: &[(&str, &str)]) -> BTreeMap<String, PersistedRoute> {
        pairs
            .iter()
            .map(|(m, p)| (m.to_string(), route(p)))
            .collect()
    }

    #[test]
    fn canonical_model_strips_context_suffix() {
        assert_eq!(canonical_model("claude-sonnet[1m]"), "claude-sonnet");
        assert_eq!(canonical_model(" claude-sonnet "), "claude-sonnet");
        assert_eq!(canonical_model("gpt-oss"), "gpt-oss");
    }

    #[test]
    fn unseeded_model_starts_at_tool_native_unconfirmed() {
        let cache = RouteCache::new("claude", ProviderProtocol::Anthropic, BTreeMap::new());
        let slot = cache.resolve("qwen3.7-max");
        assert_eq!(slot.current().0, ProviderProtocol::Anthropic);
        assert!(!slot.is_confirmed());
    }

    #[test]
    fn seeded_model_starts_confirmed_on_its_route() {
        let cache = RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            seeded(&[("gpt-oss", "openai")]),
        );
        let slot = cache.resolve("gpt-oss");
        assert_eq!(slot.current().0, ProviderProtocol::Openai);
        assert!(slot.is_confirmed());
    }

    #[test]
    fn default_route_applies_to_unlisted_models() {
        let cache = RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            seeded(&[("", "openai")]),
        );
        let slot = cache.resolve("some-new-model");
        // "" default beats tool-native, but isn't a per-model pin so stays
        // unconfirmed until the cascade proves it.
        assert_eq!(slot.current().0, ProviderProtocol::Openai);
        assert!(!slot.is_confirmed());
    }

    #[test]
    fn resolve_is_stable_across_suffix_variants() {
        let cache = RouteCache::new("claude", ProviderProtocol::Anthropic, BTreeMap::new());
        let a = cache.resolve("claude-sonnet");
        let b = cache.resolve("claude-sonnet[1m]");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn per_model_isolation_no_clobber() {
        // qwen learns Anthropic; gpt-oss learns OpenAI; neither overwrites the
        // other on the same (tool, key).
        let cache = RouteCache::new("claude", ProviderProtocol::Anthropic, BTreeMap::new());
        let qwen = cache.resolve("qwen3.7-max");
        qwen.confirm(); // confirmed on Anthropic (tool-native seed)
        let gpt = cache.resolve("gpt-oss");
        gpt.route_atom().store(
            encode_route(ProviderProtocol::Openai, PathVariant::Default),
            Ordering::Relaxed,
        );
        gpt.confirm();

        let mut dirty = cache.dirty_routes();
        dirty.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(dirty.len(), 2);
        assert_eq!(
            dirty[0],
            (
                "gpt-oss".into(),
                PersistedRoute::from_route(ProviderProtocol::Openai, PathVariant::Default)
            )
        );
        assert_eq!(
            dirty[1],
            (
                "qwen3.7-max".into(),
                PersistedRoute::from_route(ProviderProtocol::Anthropic, PathVariant::Default)
            )
        );
    }

    #[test]
    fn unconfirmed_slot_is_not_persisted() {
        let cache = RouteCache::new("claude", ProviderProtocol::Anthropic, BTreeMap::new());
        let slot = cache.resolve("qwen3.7-max");
        // Switched route but never confirmed (e.g. all candidates failed).
        slot.route_atom().store(
            encode_route(ProviderProtocol::Openai, PathVariant::Default),
            Ordering::Relaxed,
        );
        assert!(cache.dirty_routes().is_empty());
    }

    #[test]
    fn unchanged_persisted_route_is_not_rewritten() {
        let cache = RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            seeded(&[("gpt-oss", "openai")]),
        );
        let slot = cache.resolve("gpt-oss");
        slot.confirm(); // confirmed on the already-persisted OpenAI route
        assert!(cache.dirty_routes().is_empty());
    }

    #[test]
    fn changed_persisted_route_is_rewritten() {
        let cache = RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            seeded(&[("gpt-oss", "openai")]),
        );
        let slot = cache.resolve("gpt-oss");
        // Upstream drifted: this model now answers Anthropic.
        slot.route_atom().store(
            encode_route(ProviderProtocol::Anthropic, PathVariant::Default),
            Ordering::Relaxed,
        );
        slot.confirm();
        let dirty = cache.dirty_routes();
        assert_eq!(
            dirty,
            vec![(
                "gpt-oss".into(),
                PersistedRoute::from_route(ProviderProtocol::Anthropic, PathVariant::Default)
            )]
        );
    }

    #[test]
    fn garbage_protocol_in_seed_is_ignored() {
        let mut seed = BTreeMap::new();
        seed.insert("m".to_string(), route("not-a-protocol"));
        let cache = RouteCache::new("claude", ProviderProtocol::Anthropic, seed);
        let slot = cache.resolve("m");
        // Falls back to tool-native since the seed entry was unparseable.
        assert_eq!(slot.current().0, ProviderProtocol::Anthropic);
        assert!(!slot.is_confirmed());
    }

    #[test]
    fn persisted_route_round_trips_stripped_variant() {
        let pr = PersistedRoute::from_route(ProviderProtocol::Openai, PathVariant::Stripped);
        let (p, v) = decode_route(pr.to_byte().unwrap());
        assert_eq!(p, ProviderProtocol::Openai);
        assert_eq!(v, PathVariant::Stripped);
    }
}
