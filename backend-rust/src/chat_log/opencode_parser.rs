use anyhow::{bail, Context, Result};
use chrono::{TimeZone, Utc};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use tracing::{debug, error, info, warn};

use super::{AiTool, ChatMessage, ContentBlock};

use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct OpencodeState {
    pub tmux_pane: String,
    pub session_id: String,
    pub last_time_updated: i64,
    pub seen_text_lengths: HashMap<String, usize>,
    pub seen_tool_calls: HashSet<String>,
    pub seen_tool_results: HashSet<String>,
}

#[derive(Deserialize, Debug)]
struct PartData {
    #[serde(rename = "type")]
    part_type: String,

    // For text parts
    text: Option<String>,

    // For tool parts
    tool: Option<String>,
    state: Option<ToolState>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct ToolState {
    status: Option<String>,
    input: Option<Value>,
    output: Option<Value>,
}

/// Initialize the state for parsing opencode messages from a given tmux pane.
/// We match only on `tmux_pane` because OpenCode stores its own install path
/// as the session directory, not the user's working directory.
pub fn init_opencode_state_by_pane(
    db_path: &Path,
    _cwd: &Path,
    tmux_pane: &str,
) -> Result<OpencodeState> {
    let conn = Connection::open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT id, time_updated FROM session 
         WHERE tmux_pane = ?
         ORDER BY time_updated DESC LIMIT 1",
    )?;

    let result = stmt.query_row(rusqlite::params![tmux_pane], |row| {
        Ok((row.get(0)?, row.get(1)?))
    });

    let (session_id, time_updated): (String, i64) = match result {
        Ok(data) => data,
        Err(_) => {
            bail!("No opencode session found for tmux_pane {}", tmux_pane);
        }
    };

    debug!(
        "Found Opencode session: {} (updated: {}) for tmux_pane {}",
        session_id, time_updated, tmux_pane
    );

    Ok(OpencodeState {
        tmux_pane: tmux_pane.to_string(),
        session_id,
        last_time_updated: 0, // Start from beginning to get full history
        seen_text_lengths: HashMap::new(),
        seen_tool_calls: HashSet::new(),
        seen_tool_results: HashSet::new(),
    })
}
/// Fetch all new messages since the last fetch
pub fn fetch_new_messages(db_path: &Path, state: &mut OpencodeState) -> Result<Vec<ChatMessage>> {
    let conn = Connection::open(db_path)?;
    // Set a busy timeout to avoid hanging if the DB is locked by another process (like OpenCode)
    conn.busy_timeout(std::time::Duration::from_secs(5))?;

    let mut stmt = conn.prepare(
        "SELECT p.id, p.data, p.time_updated, json_extract(m.data, '$.role') 
         FROM part p
         JOIN message m ON p.message_id = m.id
         WHERE p.session_id = ? AND p.time_updated > ?
         ORDER BY p.time_updated ASC",
    )?;

    let mut rows = match stmt.query(rusqlite::params![state.session_id, state.last_time_updated]) {
        Ok(r) => r,
        Err(e) => {
            error!(
                "Failed to query opencode messages for session {}: {}",
                state.session_id, e
            );
            return Err(e.into());
        }
    };

    let mut messages = Vec::new();
    let mut new_last_time_updated = state.last_time_updated;

    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let data: String = row.get(1)?;
        let time_updated: i64 = row.get(2)?;
        let parsed_role: Option<String> = row.get(3)?;

        let role = parsed_role.unwrap_or_else(|| "assistant".to_string());

        if time_updated > new_last_time_updated {
            new_last_time_updated = time_updated;
        }

        match serde_json::from_str::<PartData>(&data) {
            Ok(part) => {
                if let Some(msg) = parse_part(&id, &part, time_updated, state, &role) {
                    messages.push(msg);
                }
            }
            Err(e) => {
                warn!(
                    "Failed to deserialize opencode part data: {} - error: {}",
                    data, e
                );
            }
        }
    }

    state.last_time_updated = new_last_time_updated;
    Ok(messages)
}

fn parse_part(
    id: &str,
    part: &PartData,
    time_updated: i64,
    state: &mut OpencodeState,
    message_role: &str,
) -> Option<ChatMessage> {
    let timestamp = Utc.timestamp_millis_opt(time_updated).single();

    let mut final_role = message_role.to_string();
    if part.part_type != "text" {
        final_role = "tool".to_string();
    }

    let block = match part.part_type.as_str() {
        "text" => {
            let full_text = part.text.as_deref().unwrap_or_default();
            let last_len = state.seen_text_lengths.get(id).copied().unwrap_or(0);

            if full_text.len() <= last_len {
                return None; // no new text to send
            }

            let new_chunk = &full_text[last_len..];
            state
                .seen_text_lengths
                .insert(id.to_string(), full_text.len());

            ContentBlock::Text {
                text: new_chunk.to_string(),
            }
        }
        "reasoning" => {
            // Extract thinking/reasoning content from AI models
            let content = part.text.as_deref().unwrap_or_default();
            let last_len = state
                .seen_text_lengths
                .get(&format!("reasoning_{}", id))
                .copied()
                .unwrap_or(0);

            if content.len() <= last_len {
                return None;
            }

            let new_chunk = &content[last_len..];
            state
                .seen_text_lengths
                .insert(format!("reasoning_{}", id), content.len());

            ContentBlock::Thinking {
                content: new_chunk.to_string(),
            }
        }
        "tool" => {
            let status = part
                .state
                .as_ref()
                .and_then(|s| s.status.as_deref())
                .unwrap_or("unknown");
            let tool_name = part.tool.clone().unwrap_or_else(|| "unknown".to_string());

            if status != "completed" {
                if !state.seen_tool_calls.insert(id.to_string()) {
                    return None; // already announced this running tool
                }

                let input = part.state.as_ref().and_then(|s| s.input.clone());
                ContentBlock::ToolCall {
                    name: tool_name.clone(),
                    summary: format!("Calling tool {}", tool_name),
                    input,
                }
            } else {
                if !state.seen_tool_results.insert(id.to_string()) {
                    return None; // already announced this completed result
                }

                let content_str = match part.state.as_ref().and_then(|s| s.output.as_ref()) {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(v) => Some(v.to_string()),
                    None => None,
                };

                ContentBlock::ToolResult {
                    tool_name: tool_name.clone(),
                    summary: format!("Tool {} finished", tool_name),
                    content: content_str,
                }
            }
        }
        _ => return None, // e.g., patch, snapshot, reasoning, etc. we skip formatting those as simple chat events for now
    };

    Some(ChatMessage {
        role: final_role,
        timestamp,
        blocks: vec![block],
    })
}
