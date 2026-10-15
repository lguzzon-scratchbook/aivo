use rusqlite::Connection;
use serde_json::{Value, json};

pub struct Message {
    pub id: String,
    pub role: String,
    pub time_created: i64,
    pub parts: Vec<Value>,
}

fn has_table(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [name],
        |_| Ok(()),
    )
    .is_ok()
}

pub fn sessions_source(conn: &Connection) -> Option<&'static str> {
    match (has_table(conn, "session_v2"), has_table(conn, "session")) {
        (true, true) => Some(
            "(SELECT id, project_id, time_updated FROM session_v2
              UNION ALL
              SELECT id, project_id, time_updated FROM session
               WHERE id NOT IN (SELECT id FROM session_v2))",
        ),
        (true, false) => Some("session_v2"),
        (false, true) => Some("session"),
        (false, false) => None,
    }
}

pub fn assistant_usage_query(conn: &Connection) -> Option<String> {
    const V2: &str = "SELECT session_id, time_created,
                json_extract(data, '$.model.id') AS model,
                json_extract(data, '$.tokens.input') AS input,
                json_extract(data, '$.tokens.output') AS output,
                json_extract(data, '$.tokens.cache.read') AS cache_read,
                json_extract(data, '$.tokens.cache.write') AS cache_write
           FROM session_message
          WHERE type = 'assistant' AND json_extract(data, '$.tokens') IS NOT NULL";
    const V1: &str = "SELECT session_id, time_created,
                json_extract(data, '$.modelID') AS model,
                json_extract(data, '$.tokens.input') AS input,
                json_extract(data, '$.tokens.output') AS output,
                json_extract(data, '$.tokens.cache.read') AS cache_read,
                json_extract(data, '$.tokens.cache.write') AS cache_write
           FROM message
          WHERE json_extract(data, '$.role') = 'assistant'
            AND json_extract(data, '$.tokens') IS NOT NULL";
    let source = match (
        has_table(conn, "session_message"),
        has_table(conn, "message"),
    ) {
        (true, true) => {
            format!("{V2} UNION ALL {V1} AND id NOT IN (SELECT id FROM session_message)")
        }
        (true, false) => V2.to_string(),
        (false, true) => V1.to_string(),
        (false, false) => return None,
    };
    Some(format!(
        "SELECT session_id, model, input, output, cache_read, cache_write
           FROM ({source})
          WHERE time_created >= COALESCE(?1, time_created)"
    ))
}

pub fn session_messages(conn: &Connection, session_id: &str) -> Vec<Message> {
    let v2 = if has_table(conn, "session_message") {
        v2_messages(conn, session_id)
    } else {
        Vec::new()
    };
    if !v2.is_empty() {
        return v2;
    }
    v1_messages(conn, session_id)
}

fn v2_messages(conn: &Connection, session_id: &str) -> Vec<Message> {
    let Ok(mut stmt) = conn.prepare(
        "SELECT id, type, time_created, data FROM session_message
          WHERE session_id = ?1 AND type IN ('user', 'assistant')
          ORDER BY seq",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([session_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    }) else {
        return Vec::new();
    };
    rows.flatten()
        .filter_map(|(id, role, time_created, data)| {
            let data: Value = serde_json::from_str(&data).ok()?;
            let parts = if role == "user" {
                let text = data.get("text").and_then(Value::as_str).unwrap_or("");
                vec![json!({"type": "text", "text": text})]
            } else {
                data.get("content")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            };
            Some(Message {
                id,
                role,
                time_created,
                parts,
            })
        })
        .collect()
}

fn v1_messages(conn: &Connection, session_id: &str) -> Vec<Message> {
    let Ok(mut stmt) = conn.prepare(
        "SELECT m.id, json_extract(m.data, '$.role'), m.time_created, p.data
           FROM message m
           JOIN part p ON p.message_id = m.id
          WHERE m.session_id = ?1
          ORDER BY m.time_created ASC, p.time_created ASC",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([session_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    }) else {
        return Vec::new();
    };
    let mut out: Vec<Message> = Vec::new();
    for (id, role, time_created, part) in rows.flatten() {
        let Ok(part) = serde_json::from_str::<Value>(&part) else {
            continue;
        };
        match out.last_mut() {
            Some(last) if last.id == id => last.parts.push(part),
            _ => out.push(Message {
                id,
                role,
                time_created,
                parts: vec![part],
            }),
        }
    }
    out
}

pub fn tool_output(state: &Value) -> String {
    if let Some(output) = state.get("output").and_then(Value::as_str) {
        return output.to_string();
    }
    state
        .get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|i| i.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

pub fn tool_error(state: &Value) -> Option<String> {
    let error = state.get("error")?;
    error
        .as_str()
        .or_else(|| error.get("message").and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_schema(conn: &Connection) {
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT);",
        )
        .unwrap();
    }

    fn v2_schema(conn: &Connection) {
        conn.execute_batch(
            "CREATE TABLE session_v2 (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
             CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT, type TEXT,
                                           seq INTEGER, time_created INTEGER, data TEXT);",
        )
        .unwrap();
    }

    fn session_ids(conn: &Connection) -> Vec<(String, i64)> {
        let src = sessions_source(conn).unwrap();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT s.id, s.time_updated FROM {src} s ORDER BY s.id"
            ))
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .flatten()
            .collect()
    }

    #[test]
    fn sessions_prefer_v2_and_keep_unmigrated_v1() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(sessions_source(&conn).is_none());
        v1_schema(&conn);
        v2_schema(&conn);
        conn.execute_batch(
            "INSERT INTO session VALUES ('a', 'p', 1), ('b', 'p', 2);
             INSERT INTO session_v2 VALUES ('a', 'p', 10), ('c', 'p', 30);",
        )
        .unwrap();
        assert_eq!(
            session_ids(&conn),
            vec![("a".into(), 10), ("b".into(), 2), ("c".into(), 30)]
        );
    }

    #[test]
    fn usage_counts_migrated_messages_once() {
        let conn = Connection::open_in_memory().unwrap();
        v1_schema(&conn);
        v2_schema(&conn);
        let tokens = r#""tokens":{"input":100,"output":5,"cache":{"read":7,"write":1}}"#;
        conn.execute_batch(&format!(
            "INSERT INTO message VALUES
               ('m1', 's', 1000, '{{\"role\":\"assistant\",\"modelID\":\"old\",{tokens}}}'),
               ('m2', 's', 2000, '{{\"role\":\"assistant\",\"modelID\":\"old\",{tokens}}}');
             INSERT INTO session_message VALUES
               ('m1', 's', 'assistant', 1, 1000, '{{\"model\":{{\"id\":\"new\"}},{tokens}}}'),
               ('m3', 's', 'assistant', 2, 3000, '{{\"model\":{{\"id\":\"new\"}},{tokens}}}'),
               ('u1', 's', 'user', 0, 900, '{{\"text\":\"hi\"}}');"
        ))
        .unwrap();
        let sql = assistant_usage_query(&conn).unwrap();
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows: Vec<(String, u64)> = stmt
            .query_map([None::<i64>], |r| Ok((r.get(1)?, r.get(2)?)))
            .unwrap()
            .flatten()
            .collect();
        let mut models: Vec<&str> = rows.iter().map(|(m, _)| m.as_str()).collect();
        models.sort();
        assert_eq!(models, ["new", "new", "old"]);

        let mut stmt = conn.prepare(&sql).unwrap();
        let after = stmt
            .query_map([Some(2500i64)], |r| r.get::<_, String>(1))
            .unwrap()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(after, ["new"]);
    }

    #[test]
    fn messages_read_v2_inline_content_in_seq_order() {
        let conn = Connection::open_in_memory().unwrap();
        v2_schema(&conn);
        conn.execute_batch(
            r#"INSERT INTO session_message VALUES
                 ('m2', 's', 'assistant', 2, 2000,
                  '{"content":[{"type":"reasoning","text":"hmm"},{"type":"text","text":"PONG"}]}'),
                 ('i', 's', 'idle', 3, 3000, '{"outcome":"succeeded"}'),
                 ('m1', 's', 'user', 1, 1000, '{"text":"ping","files":[]}');"#,
        )
        .unwrap();
        let msgs = session_messages(&conn, "s");
        let shape: Vec<(&str, usize)> = msgs
            .iter()
            .map(|m| (m.role.as_str(), m.parts.len()))
            .collect();
        assert_eq!(shape, [("user", 1), ("assistant", 2)]);
        assert_eq!(msgs[0].parts[0]["text"], "ping");
        assert_eq!(msgs[1].parts[1]["text"], "PONG");
    }

    #[test]
    fn messages_fall_back_to_v1_parts_grouped_per_message() {
        let conn = Connection::open_in_memory().unwrap();
        v1_schema(&conn);
        v2_schema(&conn);
        conn.execute_batch(
            r#"INSERT INTO message VALUES
                 ('m1', 's', 1000, '{"role":"user"}'),
                 ('m2', 's', 2000, '{"role":"assistant"}');
               INSERT INTO part VALUES
                 ('p1', 'm1', 1001, '{"type":"text","text":"ping"}'),
                 ('p2', 'm2', 2001, '{"type":"text","text":"one"}'),
                 ('p3', 'm2', 2002, '{"type":"text","text":"two"}');"#,
        )
        .unwrap();
        let msgs = session_messages(&conn, "s");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].parts.len(), 2);
    }

    #[test]
    fn tool_output_and_error_read_both_generations() {
        let v1 = json!({"status": "error", "output": "out", "error": "boom"});
        let v2 = json!({
            "status": "error",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "error": {"type": "tool.execution", "message": "boom"}
        });
        assert_eq!(tool_output(&v1), "out");
        assert_eq!(tool_output(&v2), "a\nb");
        assert_eq!(tool_error(&v1).as_deref(), Some("boom"));
        assert_eq!(tool_error(&v2).as_deref(), Some("boom"));
        assert_eq!(tool_error(&json!({"status": "completed"})), None);
    }
}
