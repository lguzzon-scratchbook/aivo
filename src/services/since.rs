use chrono::Duration;

/// Parse a relative duration like `30m`, `24h`, `7d`, `2w`.
///
/// Accepts a strictly-positive integer followed by a single suffix
/// `m`/`h`/`d`/`w` (no whitespace, no fractional values, no compound forms).
/// Returns a human-friendly error string suitable for surfacing to the CLI.
pub fn parse_since_duration(input: &str) -> Result<Duration, String> {
    let s = input.trim();
    if s.is_empty() {
        return Err("expected a duration like 7d, 24h, 30m, or 2w".into());
    }
    let last = s.chars().next_back().expect("non-empty checked above");
    let num_part = &s[..s.len() - last.len_utf8()];
    let n: i64 = num_part.parse().map_err(|_| {
        format!("invalid duration '{input}': expected a number followed by m/h/d/w")
    })?;
    if n <= 0 {
        return Err(format!("duration must be positive (got '{input}')"));
    }
    let dur = match last {
        'm' => Duration::try_minutes(n),
        'h' => Duration::try_hours(n),
        'd' => Duration::try_days(n),
        'w' => Duration::try_weeks(n),
        other => {
            return Err(format!(
                "unknown duration unit '{other}' in '{input}': use m, h, d, or w"
            ));
        }
    };
    dur.ok_or_else(|| format!("duration '{input}' is too large"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn parses_minutes_hours_days_weeks() {
        assert_eq!(parse_since_duration("30m").unwrap(), Duration::minutes(30));
        assert_eq!(parse_since_duration("24h").unwrap(), Duration::hours(24));
        assert_eq!(parse_since_duration("7d").unwrap(), Duration::days(7));
        assert_eq!(parse_since_duration("2w").unwrap(), Duration::weeks(2));
    }

    #[test]
    fn rejects_zero_negative_and_bad_units() {
        assert!(parse_since_duration("0d").is_err());
        assert!(parse_since_duration("-1d").is_err());
        assert!(parse_since_duration("7x").is_err());
        assert!(parse_since_duration("d").is_err());
        assert!(parse_since_duration("").is_err());
        assert!(parse_since_duration("7").is_err());
        assert!(parse_since_duration("7 d").is_err());
    }

    #[test]
    fn caps_overflow() {
        assert!(parse_since_duration("99999999999w").is_err());
    }

    #[test]
    fn rejects_multibyte_unit_without_panic() {
        assert!(parse_since_duration("7µ").is_err());
        assert!(parse_since_duration("7日").is_err());
        assert!(parse_since_duration("7😀").is_err());
    }
}
