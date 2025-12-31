#!/usr/bin/env python3
"""
Interactive viewer for roleplaying game conversation dataset.
Renders RPGEpisode samples in an HTML interface with navigation.
"""

import json
import tempfile
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pandas as pd


def load_dataset(path: str) -> list[dict]:
    """Load the parquet dataset and convert to list of dicts."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def format_tool_calls_html(tool_calls: list | None) -> str:
    """Format tool calls as styled HTML."""
    if not tool_calls:
        return ""

    html_parts = ['<div class="tool-calls">']
    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
        else:
            tc_id = getattr(tc, "id", "unknown")
            func = getattr(tc, "function", None)
            name = getattr(func, "name", "unknown") if func else "unknown"
            args = getattr(func, "arguments", "{}") if func else "{}"

        # Parse and pretty-print arguments
        try:
            if isinstance(args, str):
                args_obj = json.loads(args)
            else:
                args_obj = args
            args_formatted = json.dumps(args_obj, indent=2)
        except (json.JSONDecodeError, TypeError):
            args_formatted = str(args)

        html_parts.append(f"""
        <div class="tool-call">
            <div class="tool-call-header">
                <span class="tool-name">{name}</span>
                <span class="tool-id">ID: {tc_id}</span>
            </div>
            <pre class="tool-args">{args_formatted}</pre>
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def format_message_html(message: dict) -> str:
    """Format a single message as HTML."""
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")

    # Escape HTML in content
    if content:
        content = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    tool_calls_html = format_tool_calls_html(tool_calls)

    return f"""
    <div class="message-content">
        {f'<div class="content-text">{content}</div>' if content else ""}
        {tool_calls_html}
    </div>
    """


def episode_to_html(episode: dict, index: int) -> str:
    """Convert an RPGEpisode to HTML representation.

    Handles both the RPGEpisode dataclass schema and the flat parquet schema:
    - RPGEpisode: game_setting, player_character, step_count, metadata, scenario_message, actions
    - Flat parquet: game_setting, player_character, scenario, original_input, generation_model, etc.
    """
    game_setting = episode.get("game_setting", "Unknown setting")
    player_character = episode.get("player_character", "Unknown character")
    step_count = episode.get("step_count", 0)
    metadata = episode.get("metadata", {})
    scenario_message = episode.get("scenario_message", {})
    actions = episode.get("actions", [])

    # Handle flat parquet schema - build metadata from model fields
    if not metadata:
        for field in [
            "generation_model",
            "followup_model",
            "parameter_model",
            "original_input",
        ]:
            if field in episode and episode[field]:
                if field == "original_input" and isinstance(episode[field], dict):
                    metadata[field] = episode[field].get("prompt", str(episode[field]))
                else:
                    metadata[field] = episode[field]

    # Format metadata
    metadata_html = ""
    if metadata:
        metadata_items = []
        for key, value in metadata.items():
            metadata_items.append(f"<li><strong>{key}:</strong> {value}</li>")
        metadata_html = f'<ul class="metadata-list">{"".join(metadata_items)}</ul>'

    # Format actions
    actions_html = []
    for i, action in enumerate(actions):
        role = action.get("role", "unknown")
        message = action.get("message", {})

        role_class = "player-message" if role == "player" else "dm-message"
        badge_class = "player-badge" if role == "player" else "dm-badge"
        role_display = "Player" if role == "player" else "Dungeon Master"

        actions_html.append(f"""
        <div class="message {role_class}">
            <div class="message-header">
                <span class="role-badge {badge_class}">{role_display}</span>
                <span class="message-type">Turn {i + 1}</span>
            </div>
            {format_message_html(message)}
        </div>
        """)

    return f"""
    <div class="episode" id="episode-{index}">
        <div class="episode-header">
            <h2>Episode {index + 1}</h2>
            <div class="episode-stats">
                <span class="stat">Steps: {step_count}</span>
                <span class="stat">Actions: {len(actions)}</span>
            </div>
        </div>

        <div class="episode-info">
            <div class="info-card">
                <h3>Game Setting</h3>
                <p>{game_setting}</p>
            </div>
            <div class="info-card">
                <h3>Player Character</h3>
                <p>{player_character}</p>
            </div>
        </div>

        {f'<div class="metadata-section"><h3>Metadata</h3>{metadata_html}</div>' if metadata_html else ""}

        <div class="conversation-section">
            <h3>Conversation</h3>
            <div class="conversation">
                {scenario_html}
                {"".join(actions_html)}
            </div>
        </div>
    </div>
    """


def generate_html(episodes: list[dict]) -> str:
    """Generate the full HTML page with all episodes."""
    episodes_html = []
    for i, ep in enumerate(episodes):
        episodes_html.append(episode_to_html(ep, i))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Roleplaying Game Dataset Viewer</title>
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            padding: 20px 0 30px;
        }}

        h1 {{
            color: #ffd700;
            margin: 0 0 10px;
            font-size: 2rem;
        }}

        .nav-controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .nav-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .nav-btn:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}

        .nav-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .page-info {{
            font-size: 1.1rem;
            color: #a0a0a0;
        }}

        .jump-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .jump-input {{
            width: 80px;
            padding: 8px 12px;
            border: 2px solid #444;
            border-radius: 6px;
            background: #2a2a4a;
            color: #e0e0e0;
            font-size: 1rem;
            text-align: center;
        }}

        .jump-btn {{
            background: #444;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
        }}

        .jump-btn:hover {{
            background: #555;
        }}

        .episode {{
            display: none;
            background: #1e1e3f;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .episode.active {{
            display: block;
        }}

        .episode-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #3a3a5a;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        .episode-header h2 {{
            color: #ffd700;
            margin: 0;
        }}

        .episode-stats {{
            display: flex;
            gap: 15px;
        }}

        .stat {{
            background: #2a2a4a;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
        }}

        .episode-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .info-card {{
            background: #2a2a4a;
            border-radius: 10px;
            padding: 15px;
        }}

        .info-card h3 {{
            color: #8888ff;
            margin: 0 0 8px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-card p {{
            margin: 0;
            line-height: 1.5;
        }}

        .metadata-section {{
            background: #2a2a4a;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }}

        .metadata-section h3 {{
            color: #8888ff;
            margin: 0 0 10px;
            font-size: 0.9rem;
            text-transform: uppercase;
        }}

        .metadata-list {{
            margin: 0;
            padding-left: 20px;
            font-size: 0.9rem;
        }}

        .metadata-list li {{
            margin-bottom: 5px;
        }}

        .conversation-section h3 {{
            color: #8888ff;
            margin: 0 0 15px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .conversation {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}

        .message {{
            border-radius: 12px;
            padding: 15px;
            position: relative;
        }}

        .dm-message {{
            background: linear-gradient(135deg, #2d1f4e 0%, #1e1e3f 100%);
            border-left: 4px solid #9b59b6;
        }}

        .player-message {{
            background: linear-gradient(135deg, #1f3d4e 0%, #1e1e3f 100%);
            border-left: 4px solid #3498db;
        }}

        .message-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .role-badge {{
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .dm-badge {{
            background: #9b59b6;
            color: white;
        }}

        .player-badge {{
            background: #3498db;
            color: white;
        }}

        .message-type {{
            color: #888;
            font-size: 0.8rem;
        }}

        .message-content {{
            line-height: 1.6;
        }}

        .content-text {{
            margin-bottom: 10px;
        }}

        .tool-calls {{
            margin-top: 10px;
        }}

        .tool-call {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            margin-top: 8px;
            border: 1px solid #3a3a5a;
        }}

        .tool-call-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .tool-name {{
            color: #ffd700;
            font-weight: 600;
            font-family: 'Fira Code', 'Monaco', monospace;
        }}

        .tool-id {{
            color: #666;
            font-size: 0.75rem;
            font-family: monospace;
        }}

        .tool-args {{
            background: #0d0d1a;
            border-radius: 6px;
            padding: 12px;
            margin: 0;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #a0ffa0;
            font-family: 'Fira Code', 'Monaco', Consolas, monospace;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .keyboard-hint {{
            text-align: center;
            color: #666;
            font-size: 0.85rem;
            margin-top: 20px;
        }}

        kbd {{
            background: #333;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Roleplaying Game Dataset Viewer</h1>
            <p>Browse {len(episodes)} generated episodes</p>
        </header>

        <div class="nav-controls">
            <button class="nav-btn" id="prev-btn" onclick="prevEpisode()">Previous</button>
            <span class="page-info"><span id="current-page">1</span> / {len(episodes)}</span>
            <button class="nav-btn" id="next-btn" onclick="nextEpisode()">Next</button>

            <div class="jump-controls">
                <input type="number" class="jump-input" id="jump-input" min="1" max="{len(episodes)}" placeholder="#">
                <button class="jump-btn" onclick="jumpToEpisode()">Go</button>
            </div>
        </div>

        <div class="episodes-container">
            {"".join(episodes_html)}
        </div>

        <p class="keyboard-hint">Use <kbd>←</kbd> <kbd>→</kbd> arrow keys to navigate</p>
    </div>

    <script>
        const totalEpisodes = {len(episodes)};
        let currentEpisode = 0;

        function showEpisode(index) {{
            // Hide all episodes
            document.querySelectorAll('.episode').forEach(ep => ep.classList.remove('active'));

            // Show the selected episode
            const episode = document.getElementById('episode-' + index);
            if (episode) {{
                episode.classList.add('active');
                currentEpisode = index;
                document.getElementById('current-page').textContent = index + 1;

                // Update button states
                document.getElementById('prev-btn').disabled = index === 0;
                document.getElementById('next-btn').disabled = index === totalEpisodes - 1;
            }}
        }}

        function nextEpisode() {{
            if (currentEpisode < totalEpisodes - 1) {{
                showEpisode(currentEpisode + 1);
            }}
        }}

        function prevEpisode() {{
            if (currentEpisode > 0) {{
                showEpisode(currentEpisode - 1);
            }}
        }}

        function jumpToEpisode() {{
            const input = document.getElementById('jump-input');
            const pageNum = parseInt(input.value);
            if (pageNum >= 1 && pageNum <= totalEpisodes) {{
                showEpisode(pageNum - 1);
                input.value = '';
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowRight') {{
                nextEpisode();
            }} else if (e.key === 'ArrowLeft') {{
                prevEpisode();
            }}
        }});

        // Handle enter key in jump input
        document.getElementById('jump-input').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                jumpToEpisode();
            }}
        }});

        // Initialize
        showEpisode(0);
    </script>
</body>
</html>
'''


def serve_viewer(html_content: str, port: int = 8080):
    """Serve the HTML viewer on a local HTTP server."""
    # Create a temporary directory for serving
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "index.html"
        html_path.write_text(html_content)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=tmpdir, **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress logging

        server = HTTPServer(("localhost", port), Handler)
        url = f"http://localhost:{port}"

        print(f"Starting server at {url}")
        print("Press Ctrl+C to stop the server")

        # Open browser in a separate thread
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

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
        default="dataset_files/roleplaying_game_multi_step.parquet",
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
    args = parser.parse_args()

    print(f"Loading dataset from {args.path}...")
    episodes = load_dataset(args.path)
    print(f"Loaded {len(episodes)} episodes")

    print("Generating HTML viewer...")
    html_content = generate_html(episodes)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(html_content)
        print(f"Saved viewer to {output_path}")
    else:
        serve_viewer(html_content, args.port)


if __name__ == "__main__":
    main()
