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
from synthetic_data.utils import ContentBlock


def load_dataset(path: str) -> list[dict]:
    """Load the parquet dataset and convert to list of dicts."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def format_json_html(obj: dict | list | str) -> str:
    """Format a JSON object as syntax-highlighted HTML."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return f"<pre>{html.escape(obj)}</pre>"
    json_str = json.dumps(obj, indent=2, ensure_ascii=False)
    return f'<pre class="json-block">{html.escape(json_str)}</pre>'


def render_content_block(block: ContentBlock) -> str:
    """Render a single content block (text, tool_use, or tool_result)."""
    block_type = block.get("type", "text")

    if block_type == "text":
        text = block.get("text", "")
        # Escape HTML and preserve newlines
        escaped = html.escape(text).replace("\n", "<br>")
        return f'<div class="text-block">{escaped}</div>'

    elif block_type == "tool_use":
        tool_name = block.get("name", "unknown")
        tool_id = block.get("id", "")
        tool_input = block.get("input", {})
        return f"""
        <div class="tool-use-block">
            <div class="tool-header">
                <span class="tool-icon">🔧</span>
                <span class="tool-name">{html.escape(tool_name)}</span>
                <span class="tool-id">{html.escape(tool_id[:8] + "..." if len(tool_id) > 8 else tool_id)}</span>
            </div>
            <div class="tool-input">
                <div class="tool-label">Input:</div>
                {format_json_html(tool_input)}
            </div>
        </div>
        """

    elif block_type == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        tool_name = block.get("name", "unknown")
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        error_class = "error" if is_error else ""

        # Try to parse content as JSON for better formatting
        try:
            if isinstance(content, str):
                content_obj = json.loads(content)
                content_html = format_json_html(content_obj)
            else:
                content_html = format_json_html(content)
        except (json.JSONDecodeError, TypeError):
            content_html = f"<pre>{html.escape(str(content))}</pre>"

        return f"""
        <div class="tool-result-block {error_class}">
            <div class="tool-header">
                <span class="tool-icon">{"❌" if is_error else "✅"}</span>
                <span class="tool-name">{html.escape(tool_name)}</span>
                <span class="tool-id">{html.escape(tool_use_id[:8] + "..." if len(tool_use_id) > 8 else tool_use_id)}</span>
            </div>
            <div class="tool-output">
                <div class="tool-label">Result:</div>
                {content_html}
            </div>
        </div>
        """

    else:
        # Unknown block type - render as JSON
        return f'<div class="unknown-block">{format_json_html(block)}</div>'


def render_action(action: Action, index: int) -> str:
    """Render a single action from the episode."""
    role = action.get("role", "unknown")
    message = action.get("message", {})
    tool_calling_role = action.get("tool_calling_role")

    # Determine CSS class and display name based on role
    role_classes = {
        "player": ("player-action", "Player", "🎮"),
        "dungeon_master": ("dm-action", "Dungeon Master", "🎲"),
        "tool_result": ("tool-result-action", "Tool Result", "⚙️"),
    }
    css_class, display_name, icon = role_classes.get(
        role, ("unknown-action", role, "❓")
    )

    # Add info about which role called the tool (for tool_result actions)
    role_info = ""
    if tool_calling_role:
        caller_name = "Player" if tool_calling_role == "player" else "DM"
        role_info = f'<span class="tool-caller">(from {caller_name})</span>'

    # Parse content - it may be a JSON string from parquet storage
    content = message.get("content", [])
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = [{"type": "text", "text": content}]

    # Render all content blocks
    content_html = ""
    for block in content:
        content_html += render_content_block(block)

    return f'''
    <div class="action {css_class}" data-index="{index}">
        <div class="action-header">
            <span class="action-icon">{icon}</span>
            <span class="action-role">{display_name}</span>
            {role_info}
            <span class="action-index">#{index + 1}</span>
        </div>
        <div class="action-content">
            {content_html}
        </div>
    </div>
    '''


def render_episode(episode_data: RPGEpisode) -> str:
    """Render an episode dictionary as an HTML string."""
    game_setting = episode_data.get("game_setting", "Unknown")
    player_character = episode_data.get("player_character", "Unknown")
    metadata = episode_data.get("metadata", {})
    actions = episode_data.get("actions", [])

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
            <div class="actions-header">Actions ({len(actions)} total)</div>
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

    .actions-header {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-bottom: 15px;
    }

    .actions-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .action {
        border-radius: 8px;
        overflow: hidden;
    }

    .action-header {
        padding: 10px 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .action-icon { font-size: 1.1rem; }
    .action-index { margin-left: auto; color: var(--text-muted); font-size: 0.8rem; }
    .tool-caller { color: var(--text-muted); font-size: 0.8rem; font-style: italic; }

    .action-content {
        padding: 15px;
        font-size: 0.95rem;
    }

    /* Role-specific styling */
    .player-action {
        background: var(--player-bg);
        border-left: 3px solid var(--player-color);
    }
    .player-action .action-header { color: var(--player-color); }

    .dm-action {
        background: var(--dm-bg);
        border-left: 3px solid var(--dm-color);
    }
    .dm-action .action-header { color: var(--dm-color); }

    .tool-result-action {
        background: var(--tool-result-bg);
        border-left: 3px solid var(--tool-result-color);
    }
    .tool-result-action .action-header { color: var(--tool-result-color); }

    /* Content blocks */
    .text-block {
        margin-bottom: 10px;
    }

    .tool-use-block, .tool-result-block {
        background: rgba(0,0,0,0.2);
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0;
    }

    .tool-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    .tool-icon { font-size: 1rem; }
    .tool-name {
        font-weight: 600;
        color: var(--tool-color);
    }
    .tool-id {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: monospace;
    }

    .tool-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .tool-result-block.error {
        border: 1px solid var(--error-color);
        background: var(--error-bg);
    }

    .json-block {
        background: rgba(0,0,0,0.3);
        border-radius: 4px;
        padding: 10px;
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Hot reload indicator */
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


def generate_html(episodes: list[dict]) -> str:
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
    episodes: list[dict],
    port: int = 8080,
    watch_file: str | None = None,
):
    """Serve the HTML viewer with hot reload support.

    Args:
        episodes: List of episode dictionaries to display
        port: Port for the HTTP server
        watch_file: Optional file to watch for changes (triggers reload)
    """
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
        print("Press Ctrl+C to stop the server")

        # Open browser in a separate thread
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        # File watcher thread for hot reload
        if watch_file:
            watch_path = Path(watch_file)
            last_mtime = watch_path.stat().st_mtime if watch_path.exists() else 0

            def watch_for_changes():
                nonlocal last_mtime
                while True:
                    time.sleep(0.5)
                    try:
                        if watch_path.exists():
                            mtime = watch_path.stat().st_mtime
                            if mtime > last_mtime:
                                last_mtime = mtime
                                print(
                                    f"File changed: {watch_file}, triggering reload..."
                                )
                                HotReloadHandler.reload_timestamp = time.time()
                    except Exception:
                        pass

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
        default=None,
        help="File to watch for changes (enables hot reload)",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.path}...")
    episodes = load_dataset(args.path)
    print(f"Loaded {len(episodes)} episodes")

    if args.output:
        # Save to file
        html_content = generate_html(episodes)
        Path(args.output).write_text(html_content)
        print(f"Saved HTML to {args.output}")
    else:
        # Serve with HTTP
        serve_viewer(episodes, port=args.port, watch_file=args.watch)


if __name__ == "__main__":
    main()
