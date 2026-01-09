#!/usr/bin/env python3
"""
Interactive viewer for roleplaying game conversation dataset.
Renders RPGEpisode samples in an HTML interface with navigation.
"""

import html
import json
import tempfile
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pandas as pd

from synthetic_data.tasks.roleplaying import Action, RPGEpisode
from synthetic_data.utils import ContentBlock, Message


def load_dataset(path: str) -> list[RPGEpisode]:
    """Load the parquet dataset and convert to list of dicts."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")  # type: ignore[return-value]


def format_json_html(obj: object) -> str:
    """Format a JSON object as syntax-highlighted HTML."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return f"<pre>{html.escape(obj)}</pre>"
    json_str = json.dumps(obj, indent=2, ensure_ascii=False)
    return f'<pre class="json-block">{html.escape(json_str)}</pre>'


def render_content_block(block: ContentBlock | dict, block_index: int) -> str:
    """Render a single content block with full structure visibility."""
    block_type = block.get("type", "text")

    if block_type == "text":
        text = block.get("text", "")
        escaped = html.escape(text).replace("\n", "<br>")
        return f"""
        <div class="content-block text-content-block">
            <div class="block-header">
                <span class="block-type-badge text-badge">text</span>
                <span class="block-index">block {block_index}</span>
            </div>
            <div class="block-body">
                <div class="block-field">
                    <span class="field-name">text:</span>
                    <div class="field-value text-value">{escaped}</div>
                </div>
            </div>
        </div>
        """

    elif block_type == "tool_use":
        tool_name = block.get("name", "unknown")
        tool_id = block.get("id", "")
        tool_input = block.get("input", {})
        return f"""
        <div class="content-block tool-use-content-block">
            <div class="block-header">
                <span class="block-type-badge tool-use-badge">tool_use</span>
                <span class="block-index">block {block_index}</span>
            </div>
            <div class="block-body">
                <div class="block-field">
                    <span class="field-name">name:</span>
                    <span class="field-value tool-name-value">{html.escape(tool_name)}</span>
                </div>
                <div class="block-field">
                    <span class="field-name">id:</span>
                    <span class="field-value id-value">{html.escape(tool_id)}</span>
                </div>
                <div class="block-field">
                    <span class="field-name">input:</span>
                    <div class="field-value">{format_json_html(tool_input)}</div>
                </div>
            </div>
        </div>
        """

    elif block_type == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        tool_name = block.get("name", "unknown")
        content = block.get("content", "")
        is_error = block.get("is_error", False)

        # Try to parse content as JSON for better formatting
        try:
            if isinstance(content, str) and content.strip():
                content_obj = json.loads(content)
                content_html = format_json_html(content_obj)
            elif content:
                content_html = format_json_html(content)
            else:
                content_html = '<span class="empty-value">(empty)</span>'
        except (json.JSONDecodeError, TypeError):
            content_html = f"<pre>{html.escape(str(content))}</pre>"

        error_class = "error-block" if is_error else ""
        return f"""
        <div class="content-block tool-result-content-block {error_class}">
            <div class="block-header">
                <span class="block-type-badge tool-result-badge">tool_result</span>
                <span class="block-index">block {block_index}</span>
            </div>
            <div class="block-body">
                <div class="block-field">
                    <span class="field-name">name:</span>
                    <span class="field-value tool-name-value">{html.escape(tool_name)}</span>
                </div>
                <div class="block-field">
                    <span class="field-name">tool_use_id:</span>
                    <span class="field-value id-value">{html.escape(tool_use_id)}</span>
                </div>
                <div class="block-field">
                    <span class="field-name">is_error:</span>
                    <span class="field-value bool-value {"error-true" if is_error else ""}">{str(is_error).lower()}</span>
                </div>
                <div class="block-field">
                    <span class="field-name">content:</span>
                    <div class="field-value">{content_html}</div>
                </div>
            </div>
        </div>
        """

    else:
        # Unknown block type - render all fields
        return f"""
        <div class="content-block unknown-content-block">
            <div class="block-header">
                <span class="block-type-badge unknown-badge">{html.escape(block_type)}</span>
                <span class="block-index">block {block_index}</span>
            </div>
            <div class="block-body">
                {format_json_html(block)}
            </div>
        </div>
        """


def render_message(message: Message | dict) -> str:
    """Render a Message with its role and content blocks."""
    msg_role = message.get("role") or "unknown"

    # Parse content - it may be a JSON string from parquet storage
    content = message.get("content", [])
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = [{"type": "text", "text": content}]

    # Render all content blocks
    blocks_html = ""
    for i, block in enumerate(content):
        blocks_html += render_content_block(block, i)

    role_class = "user-message" if msg_role == "user" else "assistant-message"

    return f"""
    <div class="message {role_class}">
        <div class="message-header">
            <span class="message-label">Message</span>
            <span class="message-role-badge {role_class}-badge">{html.escape(msg_role)}</span>
            <span class="content-count">{len(content)} content block{"s" if len(content) != 1 else ""}</span>
        </div>
        <div class="message-content">
            <div class="content-blocks">
                {blocks_html}
            </div>
        </div>
    </div>
    """


def render_action(action: Action, index: int) -> str:
    """Render a single Action with full structure visibility."""
    role = action.get("role") or "unknown"
    message = action.get("message") or {}
    tool_calling_role = action.get("tool_calling_role")

    # Determine CSS class based on role
    role_classes = {
        "player": ("player-action", "🎮"),
        "dungeon_master": ("dm-action", "🎲"),
        "tool_result": ("tool-result-action", "⚙️"),
    }
    css_class, icon = role_classes.get(role, ("unknown-action", "❓"))

    # Build action metadata display
    metadata_items = f"""
        <div class="action-field">
            <span class="field-name">role:</span>
            <span class="field-value role-value">{html.escape(role)}</span>
        </div>
    """

    if tool_calling_role:
        metadata_items += f"""
        <div class="action-field">
            <span class="field-name">tool_calling_role:</span>
            <span class="field-value role-value">{html.escape(tool_calling_role)}</span>
        </div>
        """

    # Render the message
    message_html = render_message(message)

    return f"""
    <details class="action {css_class}" data-index="{index}" open>
        <summary class="action-header">
            <span class="action-icon">{icon}</span>
            <span class="action-label">Action</span>
            <span class="action-index">#{index}</span>
            <span class="collapse-indicator"></span>
        </summary>
        <div class="action-body">
            <div class="action-metadata">
                {metadata_items}
            </div>
            <div class="action-message-container">
                {message_html}
            </div>
        </div>
    </details>
    """


def render_episode(episode_data: RPGEpisode) -> str:
    """Render an episode dictionary as an HTML string."""
    game_setting = episode_data.get("game_setting") or "Unknown"
    player_character = episode_data.get("player_character") or "Unknown"
    metadata = episode_data.get("metadata") or {}
    actions = episode_data.get("actions") or []

    # Render metadata section
    metadata_items = ""
    for key, value in metadata.items():
        metadata_items += f"""
        <div class="metadata-item">
            <span class="metadata-key">{html.escape(str(key))}:</span>
            <span class="metadata-value">{html.escape(str(value))}</span>
        </div>
        """

    # Render all actions
    actions_html = ""
    for i, action in enumerate(actions):
        actions_html += render_action(action, i)

    return f"""
    <div class="episode">
        <div class="episode-header">
            <div class="episode-setting">
                <span class="setting-icon">🏰</span>
                <span class="setting-label">Setting:</span>
                <span class="setting-value">{html.escape(str(game_setting))}</span>
            </div>
            <div class="episode-character">
                <span class="character-icon">🧙</span>
                <span class="character-label">Player Character:</span>
                <span class="character-value">{html.escape(str(player_character))}</span>
            </div>
        </div>
        <details class="metadata-section">
            <summary>Metadata ({len(metadata)} items)</summary>
            <div class="metadata-content">
                {metadata_items}
            </div>
        </details>
        <div class="actions-section">
            <div class="actions-toolbar">
                <div class="actions-header">Actions ({len(actions)} total)</div>
                <div class="actions-controls">
                    <button class="control-btn" onclick="expandAllActions(this)">Expand All</button>
                    <button class="control-btn" onclick="collapseAllActions(this)">Collapse All</button>
                </div>
            </div>
            <div class="actions-list">
                {actions_html}
            </div>
        </div>
    </div>
    """


def generate_css() -> str:
    """Generate the CSS styles for the viewer."""
    return """
    :root {
        --bg-color: #1a1a2e;
        --card-bg: #16213e;
        --text-color: #eee;
        --text-muted: #888;
        --border-color: #0f3460;
        --player-color: #4ade80;
        --player-bg: #1a3a2a;
        --dm-color: #f472b6;
        --dm-bg: #3a1a2a;
        --tool-color: #60a5fa;
        --tool-bg: #1a2a3a;
        --tool-result-color: #fbbf24;
        --tool-result-bg: #2a2a1a;
        --json-key: #93c5fd;
        --json-string: #86efac;
        --json-number: #fcd34d;
        --error-color: #ef4444;
        --error-bg: #3a1a1a;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg-color);
        color: var(--text-color);
        line-height: 1.6;
        min-height: 100vh;
    }

    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        background: var(--card-bg);
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }

    .header h1 {
        font-size: 1.5rem;
        font-weight: 600;
    }

    .nav-controls {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-btn {
        background: var(--border-color);
        color: var(--text-color);
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        transition: background 0.2s;
    }

    .nav-btn:hover:not(:disabled) {
        background: #1a4a7e;
    }

    .nav-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .episode-counter {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .episode {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        overflow: hidden;
    }

    .episode-header {
        padding: 20px;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }

    .episode-setting, .episode-character {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .setting-icon, .character-icon {
        font-size: 1.5rem;
    }

    .setting-label, .character-label {
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .setting-value, .character-value {
        font-weight: 600;
    }

    .metadata-section {
        border-bottom: 1px solid var(--border-color);
    }

    .metadata-section summary {
        padding: 12px 20px;
        cursor: pointer;
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .metadata-content {
        padding: 0 20px 15px 20px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 8px;
    }

    .metadata-item {
        font-size: 0.85rem;
    }

    .metadata-key {
        color: var(--text-muted);
    }

    .metadata-value {
        color: var(--text-color);
    }

    .actions-section {
        padding: 20px;
    }

    .actions-toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .actions-header {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .actions-controls {
        display: flex;
        gap: 8px;
    }

    .control-btn {
        background: var(--border-color);
        color: var(--text-color);
        border: none;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: background 0.2s;
    }

    .control-btn:hover {
        background: #1a4a7e;
    }

    .actions-list {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    /* ==================== ACTION LEVEL ==================== */
    .action {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .action > summary {
        list-style: none;
        cursor: pointer;
        user-select: none;
    }

    .action > summary::-webkit-details-marker {
        display: none;
    }

    .action-header {
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .action[open] > .action-header {
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .action:not([open]) > .action-header {
        border-bottom: none;
    }

    .action-icon { font-size: 1.2rem; }
    .action-label { text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.8rem; }
    .action-index { color: var(--text-muted); font-size: 0.85rem; font-weight: normal; }

    .collapse-indicator {
        margin-left: auto;
        transition: transform 0.2s;
    }

    .collapse-indicator::after {
        content: "▼";
        font-size: 0.7rem;
        color: var(--text-muted);
    }

    .action:not([open]) .collapse-indicator::after {
        content: "▶";
    }

    .action-body {
        padding: 16px;
    }

    .action-metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px dashed rgba(255,255,255,0.1);
    }

    .action-field {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Role-specific action styling */
    .player-action {
        background: var(--player-bg);
        border-left: 4px solid var(--player-color);
    }
    .player-action .action-header { background: rgba(74, 222, 128, 0.1); color: var(--player-color); }

    .dm-action {
        background: var(--dm-bg);
        border-left: 4px solid var(--dm-color);
    }
    .dm-action .action-header { background: rgba(244, 114, 182, 0.1); color: var(--dm-color); }

    .tool-result-action {
        background: var(--tool-result-bg);
        border-left: 4px solid var(--tool-result-color);
    }
    .tool-result-action .action-header { background: rgba(251, 191, 36, 0.1); color: var(--tool-result-color); }

    /* ==================== MESSAGE LEVEL ==================== */
    .message {
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        overflow: hidden;
    }

    .message-header {
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.85rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        background: rgba(0,0,0,0.15);
    }

    .message-label {
        color: var(--text-muted);
        font-weight: 500;
    }

    .message-role-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .user-message-badge { background: #3b82f6; color: white; }
    .assistant-message-badge { background: #8b5cf6; color: white; }

    .content-count {
        margin-left: auto;
        color: var(--text-muted);
        font-size: 0.8rem;
    }

    .message-content {
        padding: 12px;
    }

    .content-blocks {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* ==================== CONTENT BLOCK LEVEL ==================== */
    .content-block {
        background: rgba(0,0,0,0.25);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .block-header {
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        background: rgba(0,0,0,0.2);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .block-type-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .text-badge { background: #22c55e; color: #000; }
    .tool-use-badge { background: #3b82f6; color: white; }
    .tool-result-badge { background: #f59e0b; color: #000; }
    .unknown-badge { background: #6b7280; color: white; }

    .block-index {
        margin-left: auto;
        color: var(--text-muted);
        font-size: 0.75rem;
    }

    .block-body {
        padding: 12px;
    }

    .block-field {
        margin-bottom: 10px;
    }

    .block-field:last-child {
        margin-bottom: 0;
    }

    /* ==================== FIELD STYLING ==================== */
    .field-name {
        color: #93c5fd;
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        font-size: 0.8rem;
        margin-right: 8px;
    }

    .field-value {
        color: var(--text-color);
    }

    .text-value {
        line-height: 1.7;
        padding: 8px 0;
    }

    .tool-name-value {
        color: #a78bfa;
        font-weight: 600;
    }

    .id-value {
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
        word-break: break-all;
    }

    .role-value {
        color: #fbbf24;
        font-weight: 500;
    }

    .bool-value {
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        color: #22c55e;
    }

    .bool-value.error-true {
        color: var(--error-color);
    }

    .empty-value {
        color: var(--text-muted);
        font-style: italic;
    }

    /* Error styling for tool results */
    .error-block {
        border-color: var(--error-color) !important;
    }
    .error-block .block-header {
        background: rgba(239, 68, 68, 0.2);
    }

    /* ==================== JSON BLOCK ==================== */
    .json-block {
        background: rgba(0,0,0,0.4);
        border-radius: 4px;
        padding: 10px;
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        font-size: 0.8rem;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
        margin-top: 4px;
    }

    /* ==================== HOT RELOAD ==================== */
    .reload-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 6px 12px;
        background: var(--tool-color);
        color: #000;
        border-radius: 4px;
        font-size: 0.8rem;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .reload-indicator.visible {
        opacity: 1;
    }
    """


def generate_html(episodes: list[RPGEpisode]) -> str:
    """Generate the full HTML page with all episodes."""
    css = generate_css()

    # Render all episodes
    episodes_html = ""
    for i, episode in enumerate(episodes):
        episode_html = render_episode(episode)
        episodes_html += f'<div class="episode-wrapper" data-episode="{i}" style="display: none;">{episode_html}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RPG Episode Viewer</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RPG Episode Viewer</h1>
            <div class="nav-controls">
                <button class="nav-btn" id="prev-btn" onclick="prevEpisode()">← Previous</button>
                <span class="episode-counter">
                    Episode <span id="current-num">1</span> of <span id="total-num">{len(episodes)}</span>
                </span>
                <button class="nav-btn" id="next-btn" onclick="nextEpisode()">Next →</button>
            </div>
        </div>
        <div id="episodes-container">
            {episodes_html}
        </div>
    </div>
    <div class="reload-indicator" id="reload-indicator">Reloading...</div>

    <script>
        let currentEpisode = 0;
        const totalEpisodes = {len(episodes)};
        let lastCheck = Date.now();

        function showEpisode(index) {{
            document.querySelectorAll('.episode-wrapper').forEach((el, i) => {{
                el.style.display = i === index ? 'block' : 'none';
            }});
            document.getElementById('current-num').textContent = index + 1;
            document.getElementById('prev-btn').disabled = index === 0;
            document.getElementById('next-btn').disabled = index === totalEpisodes - 1;
        }}

        function nextEpisode() {{
            if (currentEpisode < totalEpisodes - 1) {{
                currentEpisode++;
                showEpisode(currentEpisode);
            }}
        }}

        function prevEpisode() {{
            if (currentEpisode > 0) {{
                currentEpisode--;
                showEpisode(currentEpisode);
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === 'n') nextEpisode();
            if (e.key === 'ArrowLeft' || e.key === 'p') prevEpisode();
        }});

        // Collapse/Expand all actions in current episode
        function collapseAllActions(btn) {{
            const episode = btn.closest('.episode');
            episode.querySelectorAll('details.action').forEach(el => el.open = false);
        }}

        function expandAllActions(btn) {{
            const episode = btn.closest('.episode');
            episode.querySelectorAll('details.action').forEach(el => el.open = true);
        }}

        // Hot reload polling
        async function checkForReload() {{
            try {{
                const response = await fetch('/reload-check?t=' + lastCheck);
                const data = await response.json();
                if (data.reload) {{
                    const indicator = document.getElementById('reload-indicator');
                    indicator.classList.add('visible');
                    setTimeout(() => location.reload(), 300);
                }}
                lastCheck = data.timestamp;
            }} catch (e) {{
                // Server might be restarting, retry
            }}
            setTimeout(checkForReload, 1000);
        }}

        showEpisode(0);
        checkForReload();
    </script>
</body>
</html>"""


class HotReloadHandler(SimpleHTTPRequestHandler):
    """HTTP handler with hot reload support."""

    html_path: Path
    reload_timestamp: float
    tmpdir: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.__class__.tmpdir, **kwargs)

    def do_GET(self):
        if self.path.startswith("/reload-check"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            response = json.dumps(
                {
                    "reload": self.__class__.reload_timestamp
                    > float(self.path.split("t=")[-1])
                    if "t=" in self.path
                    else False,
                    "timestamp": self.__class__.reload_timestamp,
                }
            )
            self.wfile.write(response.encode())
        else:
            super().do_GET()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args  # Suppress logging


def serve_viewer(
    dataset_path: str,
    port: int = 8080,
    watch_files: list[str] | None = None,
):
    """Serve the HTML viewer with hot reload support.

    Args:
        dataset_path: Path to the parquet dataset file
        port: Port for the HTTP server
        watch_files: Optional additional files to watch for changes
    """
    # Load initial dataset
    episodes = load_dataset(dataset_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "index.html"
        html_content = generate_html(episodes)
        html_path.write_text(html_content)

        # Set up the handler class with shared state
        HotReloadHandler.tmpdir = tmpdir
        HotReloadHandler.html_path = html_path
        HotReloadHandler.reload_timestamp = time.time()

        server = HTTPServer(("localhost", port), HotReloadHandler)
        url = f"http://localhost:{port}"

        print(f"Starting server at {url}")
        print(f"Watching dataset: {dataset_path}")
        if watch_files:
            print(f"Watching files: {', '.join(watch_files)}")
        print("Press Ctrl+C to stop the server")

        # Open browser in a separate thread
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        # Build list of files to watch (always include dataset)
        files_to_watch = [dataset_path]
        if watch_files:
            files_to_watch.extend(watch_files)

        # Track last modification times
        last_mtimes: dict[str, float] = {}
        for f in files_to_watch:
            p = Path(f)
            last_mtimes[f] = p.stat().st_mtime if p.exists() else 0

        def watch_for_changes():
            nonlocal episodes
            while True:
                time.sleep(0.5)
                try:
                    for filepath in files_to_watch:
                        watch_path = Path(filepath)
                        if watch_path.exists():
                            mtime = watch_path.stat().st_mtime
                            if mtime > last_mtimes[filepath]:
                                last_mtimes[filepath] = mtime
                                print(f"File changed: {filepath}")

                                # If dataset changed, reload data and regenerate HTML
                                if filepath == dataset_path:
                                    print("Reloading dataset...")
                                    episodes = load_dataset(dataset_path)
                                    html_content = generate_html(episodes)
                                    html_path.write_text(html_content)
                                    print(
                                        f"Regenerated HTML with {len(episodes)} episodes"
                                    )

                                # Trigger browser reload
                                HotReloadHandler.reload_timestamp = time.time()
                except Exception as e:
                    print(f"Watch error: {e}")

        watcher = threading.Thread(target=watch_for_changes, daemon=True)
        watcher.start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="View roleplaying game dataset")
    parser.add_argument(
        "--path",
        default="dataset_files/roleplaying_game_multi_step_dev.parquet",
        help="Path to the parquet file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (if provided, saves to file instead of serving)",
    )
    parser.add_argument(
        "--watch",
        type=str,
        nargs="*",
        default=None,
        help="Additional files to watch for changes (dataset is always watched)",
    )
    args = parser.parse_args()

    if args.output:
        # Save to file
        print(f"Loading dataset from {args.path}...")
        episodes = load_dataset(args.path)
        print(f"Loaded {len(episodes)} episodes")
        html_content = generate_html(episodes)
        Path(args.output).write_text(html_content)
        print(f"Saved HTML to {args.output}")
    else:
        # Serve with HTTP (dataset watching is automatic)
        serve_viewer(args.path, port=args.port, watch_files=args.watch)


if __name__ == "__main__":
    main()
