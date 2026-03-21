use std::collections::{HashMap, HashSet};

use crate::cli::StatsArgs;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::session_store::{ApiKey, UsageCounter, UsageStats};
use crate::style;

pub struct StatsCommand {
    store: SessionStore,
}

impl StatsCommand {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }

    pub async fn execute(&self, args: StatsArgs) -> ExitCode {
        self.show(&args).await
    }

    async fn show(&self, args: &StatsArgs) -> ExitCode {
        let stats = match self.store.load_stats().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                return ExitCode::UserError;
            }
        };

        if stats.is_empty() {
            println!("{}", style::dim("No usage stats recorded yet."));
            return ExitCode::Success;
        }

        let keys = self.store.get_keys().await.unwrap_or_default();
        let fmt = if args.numbers {
            format_number
        } else {
            format_human
        };

        let all_key_ids: HashSet<&str> = keys.iter().map(|k| k.id.as_str()).collect();
        let search = args.search.as_deref().map(|s| s.to_lowercase());
        let searching = search.is_some();

        // When searching, narrow key_ids to matching keys (if any match)
        let mut key_ids = all_key_ids.clone();
        let filtering_by_key = if let Some(ref q) = search {
            let matched: HashSet<&str> = keys
                .iter()
                .filter(|k| k.name.to_lowercase().contains(q) || k.id.to_lowercase().starts_with(q))
                .map(|k| k.id.as_str())
                .collect();
            if !matched.is_empty() {
                key_ids = matched;
                true
            } else {
                false
            }
        } else {
            false
        };

        // Aggregate tools early so we can split chat vs tool runs in the summary
        let mut tool_counts = aggregate_tool_counts(&stats, &key_ids);
        let chat_runs = tool_counts.remove("chat").unwrap_or(0);

        // Summary
        println!("{}", style::bold("Usage Statistics"));
        println!();

        let (total_sel, total_prompt, total_completion, cache_read, cache_creation) =
            if filtering_by_key {
                let (mut s, mut p, mut c, mut cr, mut cc) = (0u64, 0u64, 0u64, 0u64, 0u64);
                for kid in &key_ids {
                    if let Some(e) = stats.key_usage.get(*kid) {
                        s += e.selections;
                        p += e.prompt_tokens;
                        c += e.completion_tokens;
                        cr += e.cache_read_input_tokens;
                        cc += e.cache_creation_input_tokens;
                    }
                }
                (s, p, c, cr, cc)
            } else {
                (
                    stats.total_selections,
                    stats.total_prompt_tokens,
                    stats.total_completion_tokens,
                    stats.total_cache_read_input_tokens,
                    stats.total_cache_creation_input_tokens,
                )
            };

        let total_tokens = total_prompt.saturating_add(total_completion);
        let tool_runs = total_sel.saturating_sub(chat_runs);
        let total_cache = cache_read.saturating_add(cache_creation);
        let show_cache = total_cache > 0;

        // Compute value column width for alignment
        // "Tokens:" is the longest label (7 chars), pad all to match
        let mut val_w = fmt(total_sel).len().max(fmt(total_tokens).len());
        if show_cache {
            val_w = val_w.max(fmt(total_cache).len());
        }

        println!(
            "  {} {:>w$}  {}",
            style::cyan(format!("{:<8}", "launches")),
            fmt(total_sel),
            style::dim(format!(
                "(tool {} · chat {})",
                fmt(tool_runs),
                fmt(chat_runs)
            )),
            w = val_w
        );
        println!(
            "  {} {:>w$}  {}",
            style::cyan(format!("{:<8}", "tokens")),
            fmt(total_tokens),
            style::dim(format!(
                "(prompt {} · completion {})",
                fmt(total_prompt),
                fmt(total_completion)
            )),
            w = val_w
        );
        if show_cache {
            println!(
                "  {} {:>w$}  {}",
                style::cyan(format!("{:<8}", "cached")),
                fmt(total_cache),
                style::dim(format!(
                    "(read {} · write {})",
                    fmt(cache_read),
                    fmt(cache_creation)
                )),
                w = val_w
            );
        }
        if let Some(ref q) = search {
            tool_counts.retain(|name, _| name.to_lowercase().contains(q));
        }
        if !tool_counts.is_empty() {
            println!();
            println!("{} {}", style::bold("By tool"), style::dim("(times)"));

            let mut rows: Vec<(&str, u64)> = tool_counts
                .iter()
                .map(|(name, &count)| (name.as_str(), count))
                .collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1));

            let name_w = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
            let count_w = rows.iter().map(|(_, c)| fmt(*c).len()).max().unwrap_or(0);
            let max_val = rows.iter().map(|(_, c)| *c).max().unwrap_or(0);

            for (name, count) in &rows {
                let pn = format!("{:<width$}", name, width = name_w);
                let pc = format!("{:>width$}", fmt(*count), width = count_w);
                if searching {
                    println!("  {} {}", style::cyan(&pn), pc);
                } else {
                    println!(
                        "  {} {} {}",
                        style::cyan(&pn),
                        pc,
                        style::cyan(bar(*count, max_val)),
                    );
                }
            }
        }

        // By key (only keys that still exist; filtered when searching)
        let existing_key_usage: HashMap<_, _> = stats
            .key_usage
            .iter()
            .filter(|(id, _)| key_ids.contains(id.as_str()))
            .map(|(id, counter)| (id.clone(), counter.clone()))
            .collect();
        if !existing_key_usage.is_empty() && (!searching || filtering_by_key) {
            println!();
            println!(
                "{} {}",
                style::bold("By key"),
                style::dim("(times · tokens)")
            );
            print_usage_section(
                &existing_key_usage,
                |id| key_display_name(id, &keys),
                fmt,
                !searching,
            );
        }

        // By model
        let mut model_usage = aggregate_model_usage(&stats, &key_ids);
        if let Some(ref q) = search {
            model_usage.retain(|name, _| name.to_lowercase().contains(q));
        }
        if !model_usage.is_empty() {
            println!();
            println!(
                "{} {}",
                style::bold("By model"),
                style::dim("(times · tokens)")
            );
            print_usage_section(&model_usage, |name| name.to_string(), fmt, !searching);
        }

        ExitCode::Success
    }

    pub fn print_help() {
        println!("{} aivo stats [options]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Show usage statistics: token counts, request counts, and breakdowns.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-n, --numbers", "Exact numbers instead of human-readable");
        print_opt("-s, --search <QUERY>", "Search by key, model, or tool name");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo stats"));
        println!("  {}", style::dim("aivo stats -n"));
        println!("  {}", style::dim("aivo stats -s openrouter"));
    }
}

fn print_usage_section(
    usage: &HashMap<String, UsageCounter>,
    name_fn: impl Fn(&str) -> String,
    fmt: fn(u64) -> String,
    show_bar: bool,
) {
    let mut rows: Vec<(String, u64, u64)> = usage
        .iter()
        .map(|(id, counter)| (name_fn(id), counter.selections, counter.total_tokens))
        .collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

    let name_w = rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    let sel_w = rows
        .iter()
        .map(|(_, s, _)| fmt(*s).len())
        .max()
        .unwrap_or(0);
    let tok_w = rows
        .iter()
        .map(|(_, _, t)| fmt(*t).len())
        .max()
        .unwrap_or(0);
    let max_tok = rows.iter().map(|r| r.2).max().unwrap_or(0);

    for (name, sel, tok) in &rows {
        let pn = format!("{:<width$}", name, width = name_w);
        let ps = format!("{:>width$}", fmt(*sel), width = sel_w);
        let pt = format!("{:>width$}", fmt(*tok), width = tok_w);
        let bar_str = if show_bar {
            format!(" {}", style::cyan(bar(*tok, max_tok)))
        } else {
            String::new()
        };
        println!("  {} {} {}{}", style::cyan(&pn), ps, pt, bar_str,);
    }
}

/// Aggregates tool counts from per-key data of existing keys.
/// Falls back to global tool_counts for legacy data without per-key breakdowns.
fn aggregate_tool_counts(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
) -> HashMap<String, u64> {
    let mut result: HashMap<String, u64> = HashMap::new();
    let mut has_per_key = false;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            for (tool, count) in &entry.per_tool {
                has_per_key = true;
                *result.entry(tool.clone()).or_default() += count;
            }
        }
    }
    if !has_per_key {
        return stats.tool_counts.clone();
    }
    result
}

/// Aggregates model usage from per-key data of existing keys.
/// Falls back to global model_usage for legacy data without per-key breakdowns.
fn aggregate_model_usage(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
) -> HashMap<String, UsageCounter> {
    let mut result: HashMap<String, UsageCounter> = HashMap::new();
    let mut has_per_key = false;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            for (model, sel) in &entry.per_model_selections {
                has_per_key = true;
                result.entry(model.clone()).or_default().selections += sel;
            }
            for (model, tok) in &entry.per_model_tokens {
                has_per_key = true;
                result.entry(model.clone()).or_default().total_tokens += tok;
            }
        }
    }
    if !has_per_key {
        return stats.model_usage.clone();
    }
    result
}

fn key_display_name(key_id: &str, keys: &[ApiKey]) -> String {
    keys.iter()
        .find(|k| k.id == key_id)
        .map(|k| k.display_name().to_string())
        .unwrap_or_else(|| {
            if key_id.len() > 8 {
                format!("{}…", &key_id[..8])
            } else {
                key_id.to_string()
            }
        })
}

const BAR_MAX: usize = 20;

fn bar(value: u64, max_value: u64) -> String {
    if max_value == 0 || value == 0 {
        return String::new();
    }
    let eighths = ((value as f64 / max_value as f64) * (BAR_MAX * 8) as f64).round() as usize;
    let full = eighths / 8;
    let frac = eighths % 8;
    let mut s = "█".repeat(full);
    if frac > 0 {
        s.push(
            ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"][frac]
                .chars()
                .next()
                .unwrap(),
        );
    }
    if s.is_empty() {
        s.push('▏');
    }
    s
}

fn format_number(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_human(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    if n < 1_000_000 {
        let val = n as f64 / 1_000.0;
        return if val < 10.0 {
            format!("{:.1}K", val)
        } else {
            format!("{:.0}K", val)
        };
    }
    if n < 1_000_000_000 {
        let val = n as f64 / 1_000_000.0;
        return if val < 10.0 {
            format!("{:.1}M", val)
        } else {
            format!("{:.0}M", val)
        };
    }
    let val = n as f64 / 1_000_000_000.0;
    if val < 10.0 {
        format!("{:.1}B", val)
    } else {
        format!("{:.0}B", val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_number_small() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
    }

    #[test]
    fn format_number_with_commas() {
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(12_345), "12,345");
        assert_eq!(format_number(1_234_567), "1,234,567");
        assert_eq!(format_number(1_000_000_000), "1,000,000,000");
    }

    #[test]
    fn format_human_small() {
        assert_eq!(format_human(0), "0");
        assert_eq!(format_human(42), "42");
        assert_eq!(format_human(999), "999");
    }

    #[test]
    fn format_human_thousands() {
        assert_eq!(format_human(1_000), "1.0K");
        assert_eq!(format_human(1_500), "1.5K");
        assert_eq!(format_human(9_999), "10.0K");
        assert_eq!(format_human(12_345), "12K");
        assert_eq!(format_human(998_660), "999K");
    }

    #[test]
    fn format_human_millions() {
        assert_eq!(format_human(1_000_000), "1.0M");
        assert_eq!(format_human(1_500_000), "1.5M");
        assert_eq!(format_human(12_345_678), "12M");
    }

    #[test]
    fn format_human_billions() {
        assert_eq!(format_human(1_000_000_000), "1.0B");
        assert_eq!(format_human(2_500_000_000), "2.5B");
        assert_eq!(format_human(15_000_000_000), "15B");
    }

    #[test]
    fn bar_proportional() {
        assert_eq!(bar(100, 100), "████████████████████");
        assert_eq!(bar(50, 100), "██████████");
        assert_eq!(bar(0, 100), "");
        assert_eq!(bar(0, 0), "");
    }

    #[test]
    fn bar_small_value_shows_sliver() {
        let b = bar(1, 1000);
        assert!(!b.is_empty());
        assert!(b.len() <= 4);
    }

    #[test]
    fn key_display_name_found() {
        let keys = vec![ApiKey {
            id: "abc123".to_string(),
            name: "my-key".to_string(),
            base_url: "https://api.example.com".to_string(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            key: zeroize::Zeroizing::new("secret".to_string()),
            created_at: "2024-01-01".to_string(),
        }];
        assert_eq!(key_display_name("abc123", &keys), "my-key");
    }

    #[test]
    fn key_display_name_not_found_short() {
        let keys: Vec<ApiKey> = vec![];
        assert_eq!(key_display_name("abc", &keys), "abc");
    }

    #[test]
    fn key_display_name_not_found_long() {
        let keys: Vec<ApiKey> = vec![];
        assert_eq!(key_display_name("abcdefghijklmnop", &keys), "abcdefgh…");
    }

    #[test]
    fn aggregate_tool_counts_from_per_key() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("claude".to_string(), 5);
        counter.per_tool.insert("codex".to_string(), 3);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&5));
        assert_eq!(result.get("codex"), Some(&3));
    }

    #[test]
    fn aggregate_tool_counts_falls_back_to_global() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 10);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&10));
    }

    #[test]
    fn aggregate_model_usage_from_per_key() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_model_selections.insert("gpt-4o".to_string(), 3);
        counter.per_model_tokens.insert("gpt-4o".to_string(), 1000);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_model_usage(&stats, &keys);
        assert_eq!(result.get("gpt-4o").unwrap().selections, 3);
        assert_eq!(result.get("gpt-4o").unwrap().total_tokens, 1000);
    }

    #[test]
    fn aggregate_excludes_deleted_keys() {
        let mut stats = UsageStats::default();
        let mut c1 = UsageCounter::default();
        c1.per_tool.insert("claude".to_string(), 5);
        stats.key_usage.insert("key1".to_string(), c1);
        let mut c2 = UsageCounter::default();
        c2.per_tool.insert("claude".to_string(), 3);
        stats.key_usage.insert("deleted_key".to_string(), c2);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&5));
    }
}
