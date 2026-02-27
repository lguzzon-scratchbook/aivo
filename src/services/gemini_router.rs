use anyhow::Result;
use serde_json::Value;

#[derive(Clone)]
pub struct GeminiRouterConfig {
    pub target_base_url: String,
    pub api_key: String,
}

pub struct GeminiRouter {
    config: GeminiRouterConfig,
}

impl GeminiRouter {
    pub fn new(config: GeminiRouterConfig) -> Self {
        Self { config }
    }

    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let config = self.config.clone();
        let handle = tokio::spawn(async move { run_router(listener, config).await });
        Ok((port, handle))
    }
}

async fn run_router(listener: tokio::net::TcpListener, config: GeminiRouterConfig) -> Result<()> {
    let config = std::sync::Arc::new(config);
    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = config.clone();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let request_bytes = match read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };
            let request = String::from_utf8_lossy(&request_bytes);
            let response = match handle_request(&request, &config).await {
                Ok(r) => r,
                Err(e) => {
                    let body = serde_json::json!({"error": {"message": e.to_string()}}).to_string();
                    http_response(500, "application/json", &body)
                }
            };
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

async fn handle_request(
    request: &str,
    config: &std::sync::Arc<GeminiRouterConfig>,
) -> Result<String> {
    let path = extract_path(request);

    match parse_gemini_path(&path) {
        Some((model, is_streaming)) => {
            let body_str = extract_body(request);
            let body: Value = serde_json::from_str(body_str.trim())?;
            let tool_schemas = extract_tool_schemas(&body);
            let openai_req = convert_gemini_to_openai(&body, &model);
            let openai_response = forward_to_provider(openai_req, config).await?;
            let openai_response = repair_tool_call_args(openai_response, &tool_schemas);

            if is_streaming {
                let sse = convert_openai_to_gemini_sse(&openai_response);
                Ok(http_response(200, "text/event-stream", &sse))
            } else {
                let gemini = convert_openai_to_gemini(&openai_response);
                let json = serde_json::to_string(&gemini)?;
                Ok(http_response(200, "application/json", &json))
            }
        }
        None => Ok(http_response(
            404,
            "application/json",
            "{\"error\":\"not found\"}",
        )),
    }
}

async fn forward_to_provider(
    openai_req: Value,
    config: &std::sync::Arc<GeminiRouterConfig>,
) -> Result<Value> {
    let target_url = build_chat_completions_url(&config.target_base_url);
    let client = reqwest::Client::new();
    let response = client
        .post(&target_url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&openai_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let body_text = response.text().await?;

    if status != 200 {
        anyhow::bail!("Provider error {}: {}", status, body_text);
    }

    Ok(serde_json::from_str(&body_text)?)
}

/// Constructs /v1/chat/completions URL, avoiding /v1/v1 duplication.
fn build_chat_completions_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/chat/completions", base)
    } else {
        format!("{}/v1/chat/completions", base)
    }
}

fn extract_path(request: &str) -> String {
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        "/".to_string()
    }
}

fn extract_body(request: &str) -> &str {
    request
        .find("\r\n\r\n")
        .map(|i| &request[i + 4..])
        .unwrap_or("")
        .trim_end_matches('\0')
}

fn http_response(status: u16, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        status,
        if status == 200 { "OK" } else { "Error" },
        content_type,
        body.len(),
        body
    )
}

async fn read_full_request(socket: &mut tokio::net::TcpStream) -> Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::with_capacity(16384);
    let mut tmp = vec![0u8; 4096];
    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let content_length = headers
                .lines()
                .find(|l| l.to_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body_read = buf.len() - (header_end + 4);
            if body_read < content_length {
                let remaining = content_length - body_read;
                let mut body_buf = vec![0u8; remaining];
                socket.read_exact(&mut body_buf).await?;
                buf.extend_from_slice(&body_buf);
            }
            break;
        }
    }
    Ok(buf)
}

/// Parses a Gemini API request path and extracts (model_name, is_streaming).
///
/// Examples:
/// - "/v1beta/models/gemini-2.0-flash:generateContent" → Some(("gemini-2.0-flash", false))
/// - "/v1beta/models/google/gemini-2.0-flash:streamGenerateContent?alt=sse" → Some(("google/gemini-2.0-flash", true))
/// - "/v1/chat/completions" → None
pub fn parse_gemini_path(path: &str) -> Option<(String, bool)> {
    // Strip query string
    let path = path.split('?').next().unwrap_or(path);

    let is_streaming = path.ends_with(":streamGenerateContent");
    let is_generate = path.ends_with(":generateContent");

    if !is_streaming && !is_generate {
        return None;
    }

    // Find "models/" prefix
    let models_prefix = path.find("/models/")?;
    let after_models = &path[models_prefix + "/models/".len()..];

    // Strip the trailing method suffix
    let method_suffix = if is_streaming {
        ":streamGenerateContent"
    } else {
        ":generateContent"
    };
    let model = after_models.strip_suffix(method_suffix)?;

    Some((model.to_string(), is_streaming))
}

/// Converts a Gemini generateContent request body to OpenAI chat completions format.
pub fn convert_gemini_to_openai(body: &Value, model: &str) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // System instruction → system message
    if let Some(system_text) = body
        .get("systemInstruction")
        .and_then(|si| si.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|parts| parts.first())
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
    {
        if !system_text.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": system_text}));
        }
    }

    // Convert contents → messages
    if let Some(contents) = body.get("contents").and_then(|c| c.as_array()) {
        for content in contents {
            let role = content
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("user");
            let openai_role = if role == "model" { "assistant" } else { role };
            let parts = content
                .get("parts")
                .and_then(|p| p.as_array())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            convert_parts_to_messages(parts, openai_role, &mut messages);
        }
    }

    // Convert tools
    let tools: Vec<Value> = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tool_groups| {
            tool_groups
                .iter()
                .filter_map(|tg| tg.get("functionDeclarations"))
                .filter_map(|fd| fd.as_array())
                .flatten()
                .map(|func_decl| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": func_decl.get("name").cloned().unwrap_or_default(),
                            "description": func_decl.get("description").cloned().unwrap_or_default(),
                            "parameters": normalize_parameters(func_decl.get("parameters").unwrap_or(&serde_json::json!({}))),
                        }
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let mut req = serde_json::json!({
        "model": model,
        "messages": messages,
        // Always request non-streaming from provider; for streamGenerateContent paths,
        // the router wraps the full response in a single Gemini SSE event.
        "stream": false,
    });

    if !tools.is_empty() {
        req["tools"] = Value::Array(tools);
    }

    // generationConfig → OpenAI fields
    if let Some(gc) = body.get("generationConfig") {
        if let Some(t) = gc.get("temperature") {
            req["temperature"] = t.clone();
        }
        if let Some(mt) = gc.get("maxOutputTokens") {
            req["max_tokens"] = mt.clone();
        }
        if let Some(tp) = gc.get("topP") {
            req["top_p"] = tp.clone();
        }
    }

    req
}

/// Converts Gemini content parts to one or more OpenAI messages.
/// Handles text parts, functionCall parts, and functionResponse parts.
/// Ensures a function parameters schema has `"type": "object"` at the top level.
/// Gemini CLI's built-in tools sometimes omit this, causing strict providers
/// (Vertex AI via Vercel) to reject the request with a 400 error.
/// Extracts tool parameter schemas from a Gemini request body.
/// Returns a map of function name → parameters schema.
fn extract_tool_schemas(body: &Value) -> std::collections::HashMap<String, Value> {
    let mut schemas = std::collections::HashMap::new();
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        for tg in tools {
            if let Some(decls) = tg.get("functionDeclarations").and_then(|fd| fd.as_array()) {
                for decl in decls {
                    if let (Some(name), Some(params)) = (
                        decl.get("name").and_then(|n| n.as_str()),
                        decl.get("parameters"),
                    ) {
                        schemas.insert(name.to_string(), params.clone());
                    }
                }
            }
        }
    }
    schemas
}

/// Repairs tool call arguments in an OpenAI response before converting to Gemini format.
///
/// Fixes two common model mistakes:
/// 1. Wrong parameter name (fuzzy rename: `path` → `file_path`)
/// 2. Missing required parameter with a sensible default (path-like strings → `"."`)
fn repair_tool_call_args(
    mut response: Value,
    schemas: &std::collections::HashMap<String, Value>,
) -> Value {
    if let Some(choices) = response["choices"].as_array_mut() {
        for choice in choices.iter_mut() {
            if let Some(tool_calls) = choice["message"]["tool_calls"].as_array_mut() {
                for tc in tool_calls.iter_mut() {
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    if let Some(schema) = schemas.get(&name) {
                        repair_single_tool_call(tc, schema);
                    }
                }
            }
        }
    }
    response
}

fn repair_single_tool_call(tc: &mut Value, schema: &Value) {
    let required: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if required.is_empty() {
        return;
    }

    // Parse current args (handle both string-encoded and object forms)
    let mut args: serde_json::Map<String, Value> = match &tc["function"]["arguments"] {
        Value::String(s) => serde_json::from_str(s).unwrap_or_default(),
        Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    let existing_keys: Vec<String> = args.keys().cloned().collect();

    for req in &required {
        if args.contains_key(req) {
            continue;
        }

        // 1. Fuzzy rename: find an existing key whose name overlaps with the required one
        let similar_key = existing_keys.iter().find(|k| {
            let k_lower = k.to_lowercase();
            let r_lower = req.to_lowercase();
            k_lower == r_lower || k_lower.contains(&r_lower) || r_lower.contains(&k_lower)
        });
        if let Some(old_key) = similar_key {
            if let Some(val) = args.remove(old_key) {
                args.insert(req.clone(), val);
                continue;
            }
        }

        // 2. Default: path-like string params default to current directory
        let prop_type = schema
            .get("properties")
            .and_then(|p| p.get(req))
            .and_then(|p| p.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if prop_type == "string" && (req.contains("dir") || req.ends_with("_path") || req == "path")
        {
            args.insert(req.clone(), Value::String(".".to_string()));
        }
    }

    tc["function"]["arguments"] = Value::String(
        serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".to_string()),
    );
}

fn normalize_parameters(params: &Value) -> Value {
    if let Some(obj) = params.as_object() {
        if obj.contains_key("properties") && !obj.contains_key("type") {
            let mut normalized = obj.clone();
            normalized.insert("type".to_string(), serde_json::json!("object"));
            return Value::Object(normalized);
        }
    }
    params.clone()
}

fn convert_parts_to_messages(parts: &[Value], openai_role: &str, messages: &mut Vec<Value>) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();

    for part in parts {
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            text_parts.push(text);
        } else if let Some(fc) = part.get("functionCall") {
            let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = fc
                .get("args")
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_default();
            tool_calls.push(serde_json::json!({
                "id": format!("call_{}", name),
                "type": "function",
                "function": {"name": name, "arguments": args}
            }));
        } else if let Some(fr) = part.get("functionResponse") {
            let name = fr.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let response = fr
                .get("response")
                .map(|r| serde_json::to_string(r).unwrap_or_default())
                .unwrap_or_default();
            tool_results.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": format!("call_{}", name),
                "content": response
            }));
        }
    }

    if !tool_results.is_empty() {
        // Function responses → individual tool messages
        for tr in tool_results {
            messages.push(tr);
        }
    } else if !tool_calls.is_empty() {
        // Function calls → assistant message with tool_calls
        let content_str = text_parts.join(" ");
        let mut msg = serde_json::json!({
            "role": openai_role,
            "content": if content_str.is_empty() { Value::Null } else { Value::String(content_str) },
            "tool_calls": tool_calls,
        });
        if msg["content"].is_null() {
            msg.as_object_mut().unwrap().remove("content");
        }
        messages.push(msg);
    } else {
        // Plain text message
        let content = text_parts.join("\n");
        messages.push(serde_json::json!({"role": openai_role, "content": content}));
    }
}

/// Converts an OpenAI chat completions response to Gemini generateContent response format.
pub fn convert_openai_to_gemini(body: &Value) -> Value {
    let empty_msg = serde_json::json!({"role": "assistant", "content": ""});
    let choices = body.get("choices").and_then(|c| c.as_array());
    let choice = choices
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let message = choice.get("message").cloned().unwrap_or(empty_msg);
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|r| r.as_str())
        .unwrap_or("stop");

    let gemini_finish = match finish_reason {
        "stop" | "tool_calls" => "STOP",
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "OTHER",
    };

    let parts = message_to_gemini_parts(&message);

    let candidate = serde_json::json!({
        "content": {"parts": parts, "role": "model"},
        "finishReason": gemini_finish,
        "index": 0,
    });

    let mut result = serde_json::json!({"candidates": [candidate]});

    // Usage metadata
    if let Some(usage) = body.get("usage") {
        result["usageMetadata"] = serde_json::json!({
            "promptTokenCount": usage.get("prompt_tokens").cloned().unwrap_or(Value::Null),
            "candidatesTokenCount": usage.get("completion_tokens").cloned().unwrap_or(Value::Null),
            "totalTokenCount": usage.get("total_tokens").cloned().unwrap_or(Value::Null),
        });
    }

    result
}

/// Converts an OpenAI response to a Gemini SSE stream string.
/// Returns a single SSE event with the full response.
pub fn convert_openai_to_gemini_sse(body: &Value) -> String {
    let gemini_response = convert_openai_to_gemini(body);
    let json = serde_json::to_string(&gemini_response).unwrap_or_default();
    format!("data: {}\n\n", json)
}

/// Converts an OpenAI message to Gemini parts array.
fn message_to_gemini_parts(message: &Value) -> Vec<Value> {
    // Tool calls → functionCall parts
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        return tool_calls
            .iter()
            .map(|tc| {
                let name = tc["function"]["name"].as_str().unwrap_or("");
                // Some providers return arguments as a JSON string; others as an object
                let args: Value = match &tc["function"]["arguments"] {
                    Value::String(s) => serde_json::from_str(s).unwrap_or(serde_json::json!({})),
                    obj @ Value::Object(_) => obj.clone(),
                    _ => serde_json::json!({}),
                };
                serde_json::json!({"functionCall": {"name": name, "args": args}})
            })
            .collect();
    }

    // Text content → text part
    let text = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    vec![serde_json::json!({"text": text})]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gemini_path_generate_content() {
        let result = parse_gemini_path("/v1beta/models/gemini-2.0-flash:generateContent");
        assert_eq!(result, Some(("gemini-2.0-flash".to_string(), false)));
    }

    #[test]
    fn test_parse_gemini_path_stream_generate_content() {
        let result = parse_gemini_path(
            "/v1beta/models/google/gemini-2.0-flash:streamGenerateContent?alt=sse",
        );
        assert_eq!(result, Some(("google/gemini-2.0-flash".to_string(), true)));
    }

    #[test]
    fn test_parse_gemini_path_unrecognized() {
        assert_eq!(parse_gemini_path("/v1/chat/completions"), None);
        assert_eq!(parse_gemini_path("/health"), None);
        assert_eq!(parse_gemini_path(""), None);
    }

    #[test]
    fn test_parse_gemini_path_simple_model() {
        let result = parse_gemini_path("/v1beta/models/gemini-2.5-pro:generateContent");
        assert_eq!(result, Some(("gemini-2.5-pro".to_string(), false)));
    }

    #[test]
    fn test_convert_gemini_to_openai_basic_text() {
        let body = serde_json::json!({
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi!"}]},
                {"role": "user", "parts": [{"text": "How are you?"}]}
            ]
        });
        let result = convert_gemini_to_openai(&body, "google/gemini-2.0-flash");
        assert_eq!(result["model"], "google/gemini-2.0-flash");
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Hi!");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "How are you?");
    }

    #[test]
    fn test_convert_gemini_to_openai_system_instruction() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "systemInstruction": {"parts": [{"text": "You are helpful."}]}
        });
        let result = convert_gemini_to_openai(&body, "gemini-2.0-flash");
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn test_convert_gemini_to_openai_tools() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "tools": [{"functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }]}]
        });
        let result = convert_gemini_to_openai(&body, "gemini-2.0-flash");
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_normalize_parameters_adds_type_object() {
        // Gemini CLI built-in tools often omit "type": "object" — must be added
        let params = serde_json::json!({"properties": {"path": {"type": "string"}}});
        let result = normalize_parameters(&params);
        assert_eq!(result["type"], "object");
        assert!(result["properties"].is_object());
    }

    #[test]
    fn test_normalize_parameters_preserves_existing_type() {
        let params = serde_json::json!({"type": "object", "properties": {}});
        let result = normalize_parameters(&params);
        assert_eq!(result["type"], "object");
    }

    #[test]
    fn test_convert_gemini_to_openai_tools_without_type_gets_normalized() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "tools": [{"functionDeclarations": [{
                "name": "list_directory",
                "description": "List files",
                "parameters": {"properties": {"path": {"type": "string"}}}
            }]}]
        });
        let result = convert_gemini_to_openai(&body, "gemini-2.0-flash");
        let params = &result["tools"][0]["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params["properties"].is_object());
    }

    #[test]
    fn test_convert_gemini_to_openai_generation_config() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500, "topP": 0.9}
        });
        let result = convert_gemini_to_openai(&body, "gemini-2.0-flash");
        assert_eq!(result["temperature"], 0.7);
        assert_eq!(result["max_tokens"], 500);
        assert_eq!(result["top_p"], 0.9);
    }

    #[test]
    fn test_convert_openai_to_gemini_text() {
        let response = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let result = convert_openai_to_gemini(&response);
        let candidates = result["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0]["content"]["role"], "model");
        assert_eq!(candidates[0]["content"]["parts"][0]["text"], "Hello!");
        assert_eq!(candidates[0]["finishReason"], "STOP");
        assert_eq!(result["usageMetadata"]["promptTokenCount"], 5);
        assert_eq!(result["usageMetadata"]["candidatesTokenCount"], 3);
    }

    #[test]
    fn test_convert_openai_to_gemini_tool_call() {
        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let result = convert_openai_to_gemini(&response);
        let parts = &result["candidates"][0]["content"]["parts"];
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[0]["functionCall"]["args"]["location"], "SF");
    }

    #[test]
    fn test_convert_openai_to_gemini_length_finish_reason() {
        let response = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "length"}]
        });
        let result = convert_openai_to_gemini(&response);
        assert_eq!(result["candidates"][0]["finishReason"], "MAX_TOKENS");
    }

    #[test]
    fn test_convert_openai_to_gemini_sse() {
        let response = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        });
        let sse = convert_openai_to_gemini_sse(&response);
        assert!(sse.starts_with("data: "));
        assert!(sse.contains("\"text\":\"Hi!\""));
        assert!(sse.contains("STOP"));
        // Must end with \n\n for SDK regex
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_build_chat_completions_url_with_v1() {
        assert_eq!(
            build_chat_completions_url("https://ai-gateway.vercel.sh/v1"),
            "https://ai-gateway.vercel.sh/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_chat_completions_url_without_v1() {
        assert_eq!(
            build_chat_completions_url("https://example.com"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_convert_gemini_to_openai_function_call_in_message() {
        let body = serde_json::json!({
            "contents": [
                {"role": "user", "parts": [{"text": "What's the weather?"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}}
                ]}
            ]
        });
        let result = convert_gemini_to_openai(&body, "gemini-2.0-flash");
        let messages = result["messages"].as_array().unwrap();
        // user message, assistant tool_call message, tool result message
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert!(messages[1]["tool_calls"].is_array());
        let tc = &messages[1]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_get_weather");
    }
}
