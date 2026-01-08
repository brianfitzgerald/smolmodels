#!/usr/bin/env python3
"""
Interactive viewer for roleplaying game conversation dataset.
Renders RPGEpisode samples in an HTML interface with navigation.
"""

import ast
import json
import tempfile
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data.tasks.roleplaying_tools import (
    format_tool_result_as_nl,
    format_tool_use_as_nl,
)


def load_dataset(path: str) -> list[dict]:
    """Load the parquet dataset and convert to list of dicts."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def highlight_json(json_str: str) -> str:
    """Add syntax highlighting spans to JSON string for HTML display."""
    import re

    # Escape HTML first
    json_str = json_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Highlight strings (including keys)
    json_str = re.sub(
        r'"([^"\\]*(\\.[^"\\]*)*)"',
        r'<span class="json-string">"\1"</span>',
        json_str,
    )

    # Highlight numbers
    json_str = re.sub(
        r"\b(-?\d+\.?\d*([eE][+-]?\d+)?)\b",
        r'<span class="json-number">\1</span>',
        json_str,
    )

    # Highlight booleans and null
    json_str = re.sub(
        r"\b(true|false|null)\b",
        r'<span class="json-boolean">\1</span>',
        json_str,
    )

    return json_str


def format_tool_calls_html(tool_calls: list | None, base_id: str = "") -> str:
    """Format tool calls as styled HTML with NL toggle support."""
    if not tool_calls:
        return ""

    html_parts = ['<div class="tool-calls">']
    for idx, tc in enumerate(tool_calls):
        if isinstance(tc, dict):
            tc_id = tc.get("id", f"tc-{idx}")
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
        else:
            tc_id = getattr(tc, "id", f"tc-{idx}")
            func = getattr(tc, "function", None)
            name = getattr(func, "name", "unknown") if func else "unknown"
            args = getattr(func, "arguments", "{}") if func else "{}"

        # Parse arguments
        try:
            if isinstance(args, str):
                args_obj = json.loads(args)
            else:
                args_obj = args if isinstance(args, dict) else {}
        except (json.JSONDecodeError, TypeError):
            args_obj = {}

        # Generate NL output using format_tool_use_as_nl
        nl_output = format_tool_use_as_nl(name, args_obj)

        # Pretty print and highlight the full JSON
        try:
            args_formatted = json.dumps(args_obj, indent=2)
        except (TypeError, ValueError):
            args_formatted = str(args)

        args_highlighted = highlight_json(args_formatted)

        # Generate unique block ID
        block_id = f"{base_id}-{tc_id}" if base_id else tc_id

        # Build toggle buttons and views if nl_output exists
        if nl_output:
            nl_output_escaped = (
                nl_output.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            toggle_html = f"""
            <div class="view-toggle">
                <button class="toggle-btn active" onclick="toggleToolView('{block_id}', 'nl')">Natural Language</button>
                <button class="toggle-btn" onclick="toggleToolView('{block_id}', 'json')">JSON</button>
            </div>
            """
            views_html = f"""
            <div class="nl-output-view" id="{block_id}-nl">
                <div class="nl-output-content">{nl_output_escaped}</div>
            </div>
            <div class="json-view hidden" id="{block_id}-json">
                <pre class="tool-args">{args_highlighted}</pre>
            </div>
            """
        else:
            toggle_html = ""
            views_html = f'<pre class="tool-args">{args_highlighted}</pre>'

        html_parts.append(f"""
        <div class="tool-call">
            <div class="tool-call-header">
                <span class="tool-icon">🔧</span>
                <span class="tool-name">{name}</span>
                <span class="tool-id">ID: {tc_id}</span>
            </div>
            {toggle_html}
            {views_html}
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def parse_content_blocks(content: object) -> list[dict]:
    """Parse content into a list of content block dicts, handling various formats."""
    if content is None:
        return []

    # Handle numpy arrays
    if isinstance(content, np.ndarray):
        content = content.tolist()

    # Handle stringified lists/dicts
    if isinstance(content, str):
        content = content.strip()
        if content.startswith("[") or content.startswith("{"):
            # Try json.loads first (handles escaped quotes properly)
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                # Fall back to ast.literal_eval for Python literals
                try:
                    content = ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    # Not a valid literal, treat as plain string
                    return [{"type": "text", "text": content}]
        else:
            return [{"type": "text", "text": content}]

    # Now content should be a list
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]

    return [{"type": "text", "text": str(content)}]


def format_tool_use_html(block: dict, unique_id: str = "") -> str:
    """Format a tool_use content block as styled HTML with NL toggle."""
    name = block.get("name", "unknown")
    tool_id = block.get("id", "")
    input_data = block.get("input", {})

    # Parse input data
    try:
        if isinstance(input_data, str):
            input_obj = json.loads(input_data)
        else:
            input_obj = input_data if isinstance(input_data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        input_obj = {}

    # Generate NL output using format_tool_use_as_nl
    nl_output = format_tool_use_as_nl(name, input_obj)

    # Pretty print and highlight the full JSON
    try:
        input_formatted = json.dumps(input_obj, indent=2)
    except (TypeError, ValueError):
        input_formatted = str(input_data)

    input_highlighted = highlight_json(input_formatted)

    id_html = f'<span class="tool-id">ID: {tool_id}</span>' if tool_id else ""
    block_id = unique_id or tool_id or "tool"

    # Build toggle buttons and views - always show toggle since we generate NL
    if nl_output:
        nl_output_escaped = (
            nl_output.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        toggle_html = f"""
        <div class="view-toggle">
            <button class="toggle-btn active" onclick="toggleToolView('{block_id}', 'nl')">Natural Language</button>
            <button class="toggle-btn" onclick="toggleToolView('{block_id}', 'json')">JSON</button>
        </div>
        """
        views_html = f"""
        <div class="nl-output-view" id="{block_id}-nl">
            <div class="nl-output-content">{nl_output_escaped}</div>
        </div>
        <div class="json-view hidden" id="{block_id}-json">
            <pre class="tool-args">{input_highlighted}</pre>
        </div>
        """
    else:
        toggle_html = ""
        views_html = f'<pre class="tool-args">{input_highlighted}</pre>'

    return f"""
    <div class="tool-use-block">
        <div class="tool-use-header">
            <span class="tool-icon">🔧</span>
            <span class="tool-label">Tool Call</span>
            <span class="tool-name">{name}</span>
            {id_html}
        </div>
        {toggle_html}
        {views_html}
    </div>
    """


_tool_result_counter = 0


def format_tool_result_html(block: dict) -> str:
    """Format a tool_result content block as styled HTML with NL/JSON toggle."""
    global _tool_result_counter
    tool_use_id = block.get("tool_use_id", "")
    tool_name = block.get("name", "")  # Get tool name from block
    result_content = block.get("content", "")
    is_error = block.get("is_error", False)

    # Generate unique block ID for toggle
    block_id = tool_use_id or f"tr-{_tool_result_counter}"
    _tool_result_counter += 1

    is_json = False
    result_obj = None

    # Try to parse JSON content
    try:
        if isinstance(result_content, str):
            result_content = result_content.strip()
            if result_content.startswith("{") or result_content.startswith("["):
                result_obj = json.loads(result_content)
                is_json = True
        elif isinstance(result_content, (dict, list)):
            result_obj = result_content
            is_json = True
    except (json.JSONDecodeError, TypeError):
        pass

    error_class = " tool-result-error" if is_error else ""
    id_html = f'<span class="tool-id">ID: {tool_use_id}</span>' if tool_use_id else ""
    status_icon = "❌" if is_error else "✅"
    status_text = "Error" if is_error else "Result"

    if is_json and result_obj is not None and isinstance(result_obj, dict):
        # Use tool name from block to generate NL output
        nl_output = format_tool_result_as_nl(tool_name, result_obj)

        # Create highlighted JSON view
        result_formatted = json.dumps(result_obj, indent=2)
        result_highlighted = highlight_json(result_formatted)

        # Escape NL output for HTML
        nl_output_escaped = (
            nl_output.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        toggle_html = f"""
        <div class="view-toggle">
            <button class="toggle-btn active" onclick="toggleToolView('{block_id}', 'nl')">Natural Language</button>
            <button class="toggle-btn" onclick="toggleToolView('{block_id}', 'json')">JSON</button>
        </div>
        """
        content_html = f"""
        <div class="nl-output-view" id="{block_id}-nl">
            <div class="nl-output-content">{nl_output_escaped}</div>
        </div>
        <div class="json-view hidden" id="{block_id}-json">
            <pre class="tool-result-content">{result_highlighted}</pre>
        </div>
        """
    elif is_json and result_obj is not None:
        # JSON array or non-dict - just show formatted JSON
        result_formatted = json.dumps(result_obj, indent=2)
        result_highlighted = highlight_json(result_formatted)
        toggle_html = ""
        content_html = f'<pre class="tool-result-content">{result_highlighted}</pre>'
    else:
        # Plain text result - escape HTML and display nicely
        if isinstance(result_content, str):
            result_escaped = (
                result_content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
        else:
            result_escaped = (
                str(result_content)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        toggle_html = ""
        content_html = f'<pre class="tool-result-content">{result_escaped}</pre>'

    return f"""
    <div class="tool-result-block{error_class}">
        <div class="tool-result-header">
            <span class="result-icon">{status_icon}</span>
            <span class="result-label">{status_text}</span>
            {id_html}
        </div>
        {toggle_html}
        {content_html}
    </div>
    """


_tool_use_counter = 0


def format_content_blocks_html(content: object) -> str:
    """Format content blocks as HTML, with special handling for tool_use and tool_result."""
    global _tool_use_counter
    blocks = parse_content_blocks(content)

    html_parts = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    # Escape HTML and convert newlines
                    text = (
                        text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("\n", "<br>")
                    )
                    html_parts.append(f'<div class="content-text">{text}</div>')
            elif block_type == "tool_use":
                # Generate unique ID for this tool use block
                tool_id = block.get("id", "")
                unique_id = tool_id or f"tu-{_tool_use_counter}"
                _tool_use_counter += 1
                html_parts.append(format_tool_use_html(block, unique_id))
            elif block_type == "tool_result":
                html_parts.append(format_tool_result_html(block))
            else:
                # Unknown block type, try to extract any text
                text = ""
                if "text" in block:
                    text = block["text"]
                elif "content" in block:
                    text = str(block["content"])
                if text:
                    text = (
                        text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("\n", "<br>")
                    )
                    html_parts.append(f'<div class="content-text">{text}</div>')
        elif isinstance(block, str):
            text = (
                block.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            html_parts.append(f'<div class="content-text">{text}</div>')

    return "\n".join(html_parts)


def format_message_html(message: dict) -> str:
    """Format a single message as HTML."""
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")

    # Format content blocks (handles strings, lists, numpy arrays, stringified lists)
    # This now includes proper rendering for tool_use and tool_result blocks
    content_html = format_content_blocks_html(content)

    # Format tool_calls field (separate from content blocks)
    tool_calls_html = format_tool_calls_html(tool_calls)

    return f"""
    <div class="message-content">
        {content_html}
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

    # Format scenario message
    scenario_html = ""
    if scenario_message and scenario_message.get("content"):
        scenario_html = f"""
        <div class="message dm-message">
            <div class="message-header">
                <span class="role-badge dm-badge">Dungeon Master</span>
                <span class="message-type">Scenario</span>
            </div>
            {format_message_html(scenario_message)}
        </div>
        """

    # Format actions
    actions_html = []
    for i, action in enumerate(actions):
        # Handle both old format (action IS the message) and new format (action has role + message)
        if "message" in action and isinstance(action.get("message"), dict):
            # New format: {"role": "player"|"dungeon_master", "message": {...}}
            role = action.get("role", "unknown")
            message = action.get("message", {})
        else:
            # Old format: action is the message directly
            # Infer role: DM goes first, then alternating player/DM
            # Pattern: DM, player, DM, player, DM, ...
            # Turn 0 = DM, Turn 1 = player, Turn 2 = DM, etc.
            role = "dungeon_master" if i % 2 == 0 else "player"
            message = action

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
            color: #e0e0e0;
            font-family: 'Fira Code', 'Monaco', Consolas, monospace;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        /* JSON Syntax Highlighting */
        .json-string {{
            color: #98c379;
        }}

        .json-number {{
            color: #d19a66;
        }}

        .json-boolean {{
            color: #56b6c2;
        }}

        /* Tool Use Block (in content) */
        .tool-use-block {{
            background: linear-gradient(135deg, #1a2a1a 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #3a5a3a;
        }}

        .tool-use-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .tool-icon {{
            font-size: 1rem;
        }}

        .tool-label {{
            color: #8bc34a;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Tool Result Block */
        .tool-result-block {{
            background: linear-gradient(135deg, #1a1a3a 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #3a3a6a;
        }}

        .tool-result-block.tool-result-error {{
            background: linear-gradient(135deg, #3a1a1a 0%, #1a1a2e 100%);
            border-color: #6a3a3a;
        }}

        .tool-result-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .result-icon {{
            font-size: 1rem;
        }}

        .result-label {{
            color: #64b5f6;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .tool-result-error .result-label {{
            color: #ef5350;
        }}

        .tool-result-content {{
            background: #0d0d1a;
            border-radius: 6px;
            padding: 12px;
            margin: 0;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #e0e0e0;
            font-family: 'Fira Code', 'Monaco', Consolas, monospace;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        /* View Toggle */
        .view-toggle {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .toggle-btn {{
            background: #2a2a4a;
            border: 2px solid #444;
            color: #a0a0a0;
            padding: 8px 16px;
            font-size: 0.9rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .toggle-btn:hover {{
            border-color: #667eea;
            color: #e0e0e0;
        }}

        .toggle-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: transparent;
            color: white;
        }}

        /* NL Output Display */
        .nl-output-block {{
            background: linear-gradient(135deg, #2a1f4e 0%, #1e1e3f 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #4a3a6a;
        }}

        .nl-output-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .nl-output-label {{
            color: #bb86fc;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .nl-output-content {{
            line-height: 1.6;
            color: #e0e0e0;
        }}

        .hidden {{
            display: none !important;
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

        function toggleToolView(blockId, view) {{
            const nlView = document.getElementById(blockId + '-nl');
            const jsonView = document.getElementById(blockId + '-json');

            if (!nlView || !jsonView) return;

            // Find the toggle buttons in the parent container
            const container = nlView.closest('.tool-use-block, .tool-call, .tool-result-block');
            if (!container) return;

            const buttons = container.querySelectorAll('.toggle-btn');

            if (view === 'nl') {{
                nlView.classList.remove('hidden');
                jsonView.classList.add('hidden');
                buttons[0].classList.add('active');
                buttons[1].classList.remove('active');
            }} else {{
                nlView.classList.add('hidden');
                jsonView.classList.remove('hidden');
                buttons[0].classList.remove('active');
                buttons[1].classList.add('active');
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
