#!/usr/bin/env python3
"""TUI for reviewing generated trajectory parquet files using Textual."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from scripts.trajectory_formatting import (
    normalize_tool_calls,
    normalize_value,
    parse_content_blocks,
)
from synthetic_data.utils import (
    Conversation,
    Message,
    clean_message,
    recursive_json_parse,
)

CONVERSATION_COLUMNS = ("conversation", "messages", "conversations", "trajectory")


# ---------------------------------------------------------------------------
# Color palette (GitHub-dark inspired)
# ---------------------------------------------------------------------------

ROLE_COLORS: dict[str, str] = {
    "DUNGEON_MASTER": "#e3b341",
    "PLAYER": "#7ee787",
    "ASSISTANT": "#79c0ff",
    "USER": "#d2a8ff",
    "SYSTEM": "#f0883e",
}

BLOCK_COLORS: dict[str, str] = {
    "Text": "#58a6ff",
    "Thinking": "#bc8cff",
    "Content": "#8b949e",
    "Tool Call": "#ffa657",
    "Tool Result": "#3fb950",
    "Tool Result Error": "#f85149",
}


# ---------------------------------------------------------------------------
# Plain-text rendering (test-facing interface — do not change output format)
# ---------------------------------------------------------------------------


def _append_lines(lines: list[str], content: str, indent: int) -> None:
    pad = " " * indent
    for line in content.splitlines():
        lines.append(f"{pad}{line}")


def append_content_blocks(lines: list[str], content: Any, indent: int = 0) -> None:
    """Append formatted content blocks."""
    blocks = parse_content_blocks(content)
    pad = " " * indent
    for block in blocks:
        block = normalize_value(block)
        if isinstance(block, str):
            lines.append(f"{pad}[Text]")
            _append_lines(lines, block, indent)
            continue
        if not isinstance(block, dict):
            lines.append(f"{pad}[Content]")
            lines.append(f"{pad}{preview_value(block)}")
            continue

        block_type = block.get("type", "")
        if block_type == "text":
            lines.append(f"{pad}[Text]")
            _append_lines(lines, str(block.get("text", "")), indent)
        elif block_type == "thinking":
            lines.append(f"{pad}[Thinking]")
            _append_lines(lines, str(block.get("thinking", "")), indent)
        elif block_type == "tool_use":
            name = str(block.get("name", "unknown"))
            lines.append(f"{pad}[Tool Call: {name}]")
            for line in render_arg_fields(block.get("input", {})):
                lines.append(f"{pad}  {line}")
        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            is_error = bool(block.get("is_error"))
            if tool_use_id:
                label = f"[Tool Result: {tool_use_id}]"
            else:
                label = "[Tool Result]"
            if is_error:
                label = label[:-1] + " ERROR]"
            lines.append(f"{pad}{label}")
            for line in render_fields(block.get("content", "")):
                lines.append(f"{pad}  {line}")
        else:
            lines.append(f"{pad}[{block_type or 'Content'}]")
            for line in render_fields(block):
                lines.append(f"{pad}  {line}")


def append_tool_calls(lines: list[str], tool_calls: Any, indent: int = 0) -> None:
    normalized = normalize_tool_calls(tool_calls)
    if not normalized:
        return
    pad = " " * indent
    for call in normalized:
        if isinstance(call, dict):
            fn = call.get("function", {})
            if isinstance(fn, dict):
                name = fn.get("name", "unknown")
                args = fn.get("arguments", {})
            else:
                name = "unknown"
                args = {}
            lines.append(f"{pad}[Tool Call: {name}]")
            for line in render_arg_fields(args):
                lines.append(f"{pad}  {line}")
        else:
            lines.append(f"{pad}[Tool Call]")
            lines.append(f"{pad}  {preview_value(call)}")


def append_message(
    lines: list[str], message: Message | Any, index: int, indent: int = 0
) -> None:
    message = normalize_value(message)
    pad = " " * indent
    if not isinstance(message, dict):
        lines.append(f"{pad}{index}. {message}")
        return

    role = str(message.get("role", "unknown")).upper()
    lines.append(f"{pad}{index}. [{role}]")
    body_indent = indent + 2
    content = message.get("content", "")
    if content:
        append_content_blocks(lines, content, indent=body_indent)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        append_tool_calls(lines, tool_calls, indent=body_indent)


def render_fields(obj: Any) -> list[str]:
    obj = normalize_value(obj)
    if isinstance(obj, dict):
        return [f"{key}: {preview_value(value)}" for key, value in obj.items()]
    if isinstance(obj, list):
        return [preview_value(item) for item in obj]
    return [str(obj)]


def parse_json_if_string(value: Any) -> Any:
    parsed = recursive_json_parse(value)
    return normalize_value(parsed)


def render_arg_fields(args: Any) -> list[str]:
    args = parse_json_if_string(normalize_value(args))
    if isinstance(args, dict):
        lines: list[str] = []
        for key, value in args.items():
            rendered = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=True)
            )
            lines.append(f"{key}: {rendered}")
        return lines
    return render_fields(args)


def get_messages_from_row(row: dict[str, Any]) -> Conversation:
    for key in CONVERSATION_COLUMNS:
        if key in row:
            raw = normalize_value(row[key])
            if isinstance(raw, list):
                if raw and isinstance(raw[0], list):
                    nested = normalize_value(raw[0])
                    if isinstance(nested, list):
                        return [m for m in nested if isinstance(m, dict)]
                return [m for m in raw if isinstance(m, dict)]
    return []


def detect_row_type(row: dict[str, Any]) -> str:
    if "actions" in row:
        return "roleplay"
    if get_messages_from_row(row):
        return "conversation"
    return "generic"


def preview_value(value: Any) -> str:
    value = normalize_value(value)
    if isinstance(value, str):
        return clean_message(value, truncate_length=None)
    if isinstance(value, (dict, list, bool, int, float)):
        return clean_message(value, truncate_length=None)
    return str(value)


def render_roleplay_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("== Episode ==")
    lines.append(f"game_setting: {row.get('game_setting', '')}")
    lines.append(f"player_character: {row.get('player_character', '')}")
    lines.append(f"step_count: {row.get('step_count', '')}")
    lines.append("")

    metadata = normalize_value(row.get("metadata"))
    if isinstance(metadata, dict) and metadata:
        lines.append("== Metadata ==")
        for key, value in metadata.items():
            lines.append(f"{key}: {preview_value(value)}")
        lines.append("")

    metrics = normalize_value(row.get("metrics"))
    if isinstance(metrics, dict) and metrics:
        lines.append("== Metrics ==")
        for key, value in metrics.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    actions = normalize_value(row.get("actions")) or []
    if not isinstance(actions, list):
        actions = [actions]
    lines.append("== Actions ==")
    for i, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            lines.append(f"{i}. {action}")
            lines.append("")
            continue

        role = str(action.get("role", "unknown")).upper()
        lines.append(f"{i}. [{role}]")

        action_messages = normalize_value(action.get("messages")) or []
        if not isinstance(action_messages, list):
            action_messages = [action_messages]

        if action_messages:
            for j, message in enumerate(action_messages, start=1):
                append_message(lines, message, j, indent=2)
        else:
            lines.append("  No messages")
        lines.append("")

    return lines


def render_conversation_lines(row: dict[str, Any]) -> list[str]:
    messages: Conversation = get_messages_from_row(row)
    lines: list[str] = ["== Conversation =="]
    for i, message in enumerate(messages, start=1):
        append_message(lines, message, i)
        lines.append("")
    if not messages:
        lines.append("No conversation messages detected.")
    return lines


def render_generic_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = ["== Row Fields =="]
    for key, value in row.items():
        lines.append(f"{key}: {preview_value(value)}")
    return lines


def render_row_text(row: dict[str, Any]) -> str:
    row_type = detect_row_type(row)
    if row_type == "roleplay":
        body_lines = render_roleplay_lines(row)
    elif row_type == "conversation":
        body_lines = render_conversation_lines(row)
    else:
        body_lines = render_generic_lines(row)

    return "\n".join(body_lines) if body_lines else ""


# ---------------------------------------------------------------------------
# Rich renderable pipeline (produces Rich objects for RichLog)
# ---------------------------------------------------------------------------


def _kv_table(pairs: list[tuple[str, str]]) -> Table:
    """Build a compact key-value table with no header."""
    t = Table(show_header=False, show_edge=False, box=None, padding=(0, 1, 0, 0))
    t.add_column("key", style="#8b949e", no_wrap=True)
    t.add_column("value", style="#e6edf3")
    for k, v in pairs:
        t.add_row(k, v)
    return t


def render_content_block_rich(block: Any) -> list[Any]:
    """Render a single content block to a list of Rich renderables."""
    block = normalize_value(block)
    parts: list[Any] = []

    if isinstance(block, str):
        parts.append(Text("\u25b8 TEXT", style=f"bold {BLOCK_COLORS['Text']}"))
        parts.append(Text(block, style="#c9d1d9"))
        return parts

    if not isinstance(block, dict):
        parts.append(Text("\u25b8 CONTENT", style=f"bold {BLOCK_COLORS['Content']}"))
        parts.append(Text(preview_value(block), style="#8b949e"))
        return parts

    block_type = block.get("type", "")

    if block_type == "text":
        parts.append(Text("\u25b8 TEXT", style=f"bold {BLOCK_COLORS['Text']}"))
        parts.append(Text(str(block.get("text", "")), style="#c9d1d9"))

    elif block_type == "thinking":
        parts.append(
            Text("\u25b8 THINKING", style=f"bold italic {BLOCK_COLORS['Thinking']}")
        )
        parts.append(Text(str(block.get("thinking", "")), style="italic #6e7681"))

    elif block_type == "tool_use":
        name = str(block.get("name", "unknown"))
        label = Text()
        label.append("\u25b8 TOOL CALL", style=f"bold {BLOCK_COLORS['Tool Call']}")
        label.append(f"  {name}", style=f"bold {BLOCK_COLORS['Tool Call']}")
        parts.append(label)
        args = parse_json_if_string(normalize_value(block.get("input", {})))
        if isinstance(args, dict) and args:
            pairs = []
            for k, v in args.items():
                rendered = v if isinstance(v, str) else json.dumps(v, ensure_ascii=True)
                pairs.append((k, rendered))
            parts.append(Padding(_kv_table(pairs), (0, 0, 0, 2)))
        elif args:
            parts.append(Padding(Text(str(args), style="#c9d1d9"), (0, 0, 0, 2)))

    elif block_type == "tool_result":
        tool_use_id = block.get("tool_use_id")
        is_error = bool(block.get("is_error"))
        color = (
            BLOCK_COLORS["Tool Result Error"]
            if is_error
            else BLOCK_COLORS["Tool Result"]
        )
        label_text = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
        label = Text()
        label.append(f"\u25b8 {label_text}", style=f"bold {color}")
        if tool_use_id:
            label.append(f"  {tool_use_id}", style="#8b949e")
        parts.append(label)
        content = block.get("content", "")
        content = normalize_value(content)
        if isinstance(content, dict):
            pairs = [(k, preview_value(v)) for k, v in content.items()]
            parts.append(Padding(_kv_table(pairs), (0, 0, 0, 2)))
        elif isinstance(content, str) and content:
            # Try to parse as JSON for nicer display
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    pairs = [(k, preview_value(v)) for k, v in parsed.items()]
                    parts.append(Padding(_kv_table(pairs), (0, 0, 0, 2)))
                else:
                    parts.append(Padding(Text(content, style="#c9d1d9"), (0, 0, 0, 2)))
            except (json.JSONDecodeError, ValueError):
                parts.append(Padding(Text(content, style="#c9d1d9"), (0, 0, 0, 2)))
        elif content:
            parts.append(Padding(Text(str(content), style="#c9d1d9"), (0, 0, 0, 2)))

    else:
        label_str = block_type.upper() if block_type else "CONTENT"
        parts.append(
            Text(f"\u25b8 {label_str}", style=f"bold {BLOCK_COLORS['Content']}")
        )
        for line in render_fields(block):
            parts.append(Padding(Text(line, style="#c9d1d9"), (0, 0, 0, 2)))

    return parts


def render_tool_calls_rich(tool_calls: Any) -> list[Any]:
    """Render tool_calls field to Rich renderables."""
    normalized = normalize_tool_calls(tool_calls)
    if not normalized:
        return []
    parts: list[Any] = []
    for call in normalized:
        if isinstance(call, dict):
            fn = call.get("function", {})
            if isinstance(fn, dict):
                name = fn.get("name", "unknown")
                args = fn.get("arguments", {})
            else:
                name = "unknown"
                args = {}
            label = Text()
            label.append("\u25b8 TOOL CALL", style=f"bold {BLOCK_COLORS['Tool Call']}")
            label.append(f"  {name}", style=f"bold {BLOCK_COLORS['Tool Call']}")
            parts.append(label)
            args = parse_json_if_string(normalize_value(args))
            if isinstance(args, dict) and args:
                pairs = []
                for k, v in args.items():
                    rendered = (
                        v if isinstance(v, str) else json.dumps(v, ensure_ascii=True)
                    )
                    pairs.append((k, rendered))
                parts.append(Padding(_kv_table(pairs), (0, 0, 0, 2)))
        else:
            parts.append(
                Text("\u25b8 TOOL CALL", style=f"bold {BLOCK_COLORS['Tool Call']}")
            )
            parts.append(
                Padding(Text(preview_value(call), style="#c9d1d9"), (0, 0, 0, 2))
            )
    return parts


def render_message_rich(message: Message | Any, index: int) -> list[Any]:
    """Render a single message to a list of Rich renderables."""
    message = normalize_value(message)
    parts: list[Any] = []

    if not isinstance(message, dict):
        parts.append(Text(f"{index}. {message}", style="#c9d1d9"))
        return parts

    role = str(message.get("role", "unknown")).upper()
    color = ROLE_COLORS.get(role, "#c9d1d9")

    label = Text()
    label.append(f"{index}. ", style="#484f58")
    label.append(f"[{role}]", style=f"bold {color}")
    parts.append(label)

    content = message.get("content", "")
    if content:
        blocks = parse_content_blocks(content)
        for block in blocks:
            for r in render_content_block_rich(block):
                parts.append(Padding(r, (0, 0, 0, 4)))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        for r in render_tool_calls_rich(tool_calls):
            parts.append(Padding(r, (0, 0, 0, 4)))

    return parts


def render_action_rich(action: Any, index: int) -> list[Any]:
    """Render a single action (with role badge and messages) to Rich renderables."""
    parts: list[Any] = []

    if not isinstance(action, dict):
        parts.append(Text(f"{index}. {action}", style="#c9d1d9"))
        parts.append(Text(""))
        return parts

    role = str(action.get("role", "unknown")).upper()
    color = ROLE_COLORS.get(role, "#c9d1d9")

    badge = Text()
    badge.append(f"{index}.  ", style="#484f58")
    badge.append(f" {role} ", style=f"bold #0d1117 on {color}")
    parts.append(badge)

    action_messages = normalize_value(action.get("messages")) or []
    if not isinstance(action_messages, list):
        action_messages = [action_messages]

    if action_messages:
        for j, msg in enumerate(action_messages, start=1):
            for r in render_message_rich(msg, j):
                parts.append(Padding(r, (0, 0, 0, 2)))
    else:
        parts.append(Padding(Text("No messages", style="#6e7681"), (0, 0, 0, 2)))

    parts.append(Text(""))
    return parts


def render_episode_section(row: dict[str, Any]) -> Panel:
    """Render Episode info as a Rich Panel."""
    pairs = [
        ("game_setting", str(row.get("game_setting", ""))),
        ("player_character", str(row.get("player_character", ""))),
        ("step_count", str(row.get("step_count", ""))),
    ]
    return Panel(
        _kv_table(pairs),
        title="[bold #e6edf3]Episode[/]",
        border_style="#30363d",
        padding=(0, 1),
    )


def render_metadata_section(row: dict[str, Any]) -> Panel | None:
    """Render Metadata as a Rich Panel, or None if empty."""
    metadata = normalize_value(row.get("metadata"))
    if not isinstance(metadata, dict) or not metadata:
        return None
    pairs = [(k, preview_value(v)) for k, v in metadata.items()]
    return Panel(
        _kv_table(pairs),
        title="[bold #e6edf3]Metadata[/]",
        border_style="#30363d",
        padding=(0, 1),
    )


def render_metrics_section(row: dict[str, Any]) -> Panel | None:
    """Render Metrics as a Rich Panel, or None if empty."""
    metrics = normalize_value(row.get("metrics"))
    if not isinstance(metrics, dict) or not metrics:
        return None
    pairs = [(k, str(v)) for k, v in metrics.items()]
    return Panel(
        _kv_table(pairs),
        title="[bold #e6edf3]Metrics[/]",
        border_style="#30363d",
        padding=(0, 1),
    )


def render_roleplay_rich(row: dict[str, Any]) -> list[Any]:
    """Render a roleplay row to a list of Rich renderables."""
    parts: list[Any] = []

    parts.append(render_episode_section(row))
    parts.append(Text(""))

    meta_panel = render_metadata_section(row)
    if meta_panel:
        parts.append(meta_panel)
        parts.append(Text(""))

    metrics_panel = render_metrics_section(row)
    if metrics_panel:
        parts.append(metrics_panel)
        parts.append(Text(""))

    parts.append(Rule("Actions", style="#58a6ff"))
    parts.append(Text(""))

    actions = normalize_value(row.get("actions")) or []
    if not isinstance(actions, list):
        actions = [actions]
    for i, action in enumerate(actions, start=1):
        parts.extend(render_action_rich(action, i))

    return parts


def render_conversation_rich(row: dict[str, Any]) -> list[Any]:
    """Render a conversation row to Rich renderables."""
    messages: Conversation = get_messages_from_row(row)
    parts: list[Any] = []
    parts.append(Rule("Conversation", style="#58a6ff"))
    parts.append(Text(""))
    if not messages:
        parts.append(Text("No conversation messages detected.", style="#6e7681"))
        return parts
    for i, message in enumerate(messages, start=1):
        parts.extend(render_message_rich(message, i))
        parts.append(Text(""))
    return parts


def render_generic_rich(row: dict[str, Any]) -> list[Any]:
    """Render a generic row to Rich renderables."""
    pairs = [(k, preview_value(v)) for k, v in row.items()]
    return [
        Panel(
            _kv_table(pairs),
            title="[bold #e6edf3]Row Fields[/]",
            border_style="#30363d",
            padding=(0, 1),
        )
    ]


def render_row_rich(row: dict[str, Any]) -> list[Any]:
    """Render a row to a list of Rich renderables based on its type."""
    row_type = detect_row_type(row)
    if row_type == "roleplay":
        return render_roleplay_rich(row)
    elif row_type == "conversation":
        return render_conversation_rich(row)
    else:
        return render_generic_rich(row)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def list_parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)


def resolve_default_path(dataset_dir: Path) -> Path:
    parquet_files = list_parquet_files(dataset_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    return parquet_files[-1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Textual app — Screen-based architecture
# ---------------------------------------------------------------------------

_SEP = "  [#30363d]\u2502[/]  "

APP_CSS = """
Screen {
    layout: vertical;
    background: #0d1117;
}

/* ---- ViewerScreen ---- */

ViewerScreen #viewer-header {
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #e6edf3;
    text-style: bold;
}

ViewerScreen RichLog {
    height: 1fr;
    width: 100%;
    background: #0d1117;
    padding: 0 1;
    scrollbar-color: #484f58;
    scrollbar-color-hover: #8b949e;
    scrollbar-color-active: #c9d1d9;
    scrollbar-background: #0d1117;
    scrollbar-background-hover: #161b22;
}

ViewerScreen #viewer-status {
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #8b949e;
}

ViewerScreen Footer {
    background: #0d1117;
    color: #6e7681;
}

/* ---- PickerScreen ---- */

PickerScreen #picker-header {
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #e6edf3;
    text-style: bold;
}

PickerScreen OptionList {
    height: 1fr;
    width: 100%;
    background: #0d1117;
    color: #c9d1d9;
    scrollbar-color: #484f58;
    scrollbar-color-hover: #8b949e;
    scrollbar-color-active: #c9d1d9;
    scrollbar-background: #0d1117;
    scrollbar-background-hover: #161b22;
}

PickerScreen OptionList > .option-list--option-highlighted {
    background: #58a6ff;
    color: #0d1117;
    text-style: bold;
}

PickerScreen #picker-status {
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #8b949e;
}

PickerScreen Footer {
    background: #0d1117;
    color: #6e7681;
}
"""


class PickerScreen(Screen):
    """File selection screen using OptionList."""

    BINDINGS = [
        Binding("j", "cursor_down", "Next", show=False),
        Binding("k", "cursor_up", "Prev", show=False),
        Binding("g", "first", "First", show=True),
        Binding("shift+g", "last", "Last", show=True),
        Binding("enter", "select_file", "Select", show=True),
        Binding("q", "cancel", "Cancel", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        files: list[Path],
        dataset_dir: Path,
        prefer_latest: bool = True,
    ) -> None:
        super().__init__()
        self.files = files
        self.dataset_dir = dataset_dir
        self.prefer_latest = prefer_latest

    def compose(self) -> ComposeResult:
        yield Static("", id="picker-header", markup=True)
        yield OptionList(id="picker-list")
        yield Static("", id="picker-status", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._populate_list()
        self._update_header()
        self._update_status()

    def _populate_list(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        for fp in self.files:
            stat = fp.stat()
            modified = str(pd.Timestamp.fromtimestamp(stat.st_mtime))
            label = Text()
            label.append(fp.name, style="bold #e6edf3")
            label.append("  \u2502  ", style="#30363d")
            label.append(modified, style="#8b949e")
            label.append("  \u2502  ", style="#30363d")
            label.append(f"{stat.st_size:,} B", style="#6e7681")
            ol.add_option(Option(label, id=str(fp)))
        if self.files:
            start_idx = len(self.files) - 1 if self.prefer_latest else 0
            ol.highlighted = start_idx

    def _update_header(self) -> str:
        ol = self.query_one("#picker-list", OptionList)
        idx = (ol.highlighted or 0) + 1
        total = len(self.files)
        markup = (
            "[bold #e6edf3]Trajectory Viewer[/]"
            f"{_SEP}"
            "[#58a6ff]Select File[/]"
            f"{_SEP}"
            f"[bold #3fb950]{idx}/{total}[/]"
        )
        self.query_one("#picker-header", Static).update(markup)
        return markup

    def _update_status(self) -> None:
        self.query_one("#picker-status", Static).update(
            f"[#6e7681]Dir[/] [#58a6ff]{escape(str(self.dataset_dir))}[/]"
        )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:  # noqa: ARG002
        self._update_header()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        path = Path(str(event.option_id))
        self.dismiss(path)

    def action_cursor_down(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        ol.action_cursor_down()

    def action_cursor_up(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        ol.action_cursor_up()

    def action_first(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        ol.highlighted = 0

    def action_last(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        ol.highlighted = len(self.files) - 1

    def action_select_file(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        if ol.highlighted is not None and self.files:
            path = self.files[ol.highlighted]
            self.dismiss(path)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ViewerScreen(Screen):
    """Trajectory viewer screen using RichLog."""

    BINDINGS = [
        Binding("j", "scroll_down", "Scroll Down", show=False),
        Binding("k", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("n", "next_row", "Next Row", show=True),
        Binding("shift+n", "prev_row", "Prev Row", show=True),
        Binding("g", "first_row", "First Row", show=True),
        Binding("shift+g", "last_row", "Last Row", show=True),
        Binding("o", "open_picker", "Open File", show=True),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("", id="viewer-header", markup=True)
        yield RichLog(markup=True, wrap=True, id="viewer-body")
        yield Static("", id="viewer-status", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.render_current_row()

    @property
    def app_state(self) -> TrajectoryTuiApp:
        return self.app  # type: ignore[return-value]

    def render_current_row(self) -> None:
        app = self.app_state
        if not app.rows:
            return
        row = {k: normalize_value(v) for k, v in app.rows[app.idx].items()}
        row_type = detect_row_type(row)
        renderables = render_row_rich(row)

        body = self.query_one("#viewer-body", RichLog)
        body.clear()
        for r in renderables:
            body.write(r)

        self._update_header(row_type)
        self._update_status()

    def _update_header(self, row_type: str) -> None:
        app = self.app_state
        name = app.path.name if app.path is not None else ""
        markup = (
            "[bold #e6edf3]Trajectory Viewer[/]"
            f"{_SEP}"
            f"[#58a6ff]{escape(name)}[/]"
            f"{_SEP}"
            f"[bold #3fb950]{app.idx + 1}/{len(app.rows)}[/]"
            f"{_SEP}"
            f"[#bc8cff]{escape(row_type)}[/]"
        )
        self.query_one("#viewer-header", Static).update(markup)

    def _update_status(self) -> None:
        app = self.app_state
        self.query_one("#viewer-status", Static).update(
            f"[#6e7681]Row[/] "
            f"[bold #c9d1d9]{app.idx + 1}[/]"
            f"[#484f58]/[/]"
            f"[#8b949e]{len(app.rows)}[/]"
        )

    def action_scroll_down(self) -> None:
        body = self.query_one("#viewer-body", RichLog)
        body.scroll_down(animate=False)

    def action_scroll_up(self) -> None:
        body = self.query_one("#viewer-body", RichLog)
        body.scroll_up(animate=False)

    def action_next_row(self) -> None:
        app = self.app_state
        if app.idx < len(app.rows) - 1:
            app.idx += 1
            self.render_current_row()

    def action_prev_row(self) -> None:
        app = self.app_state
        if app.idx > 0:
            app.idx -= 1
            self.render_current_row()

    def action_first_row(self) -> None:
        app = self.app_state
        if app.idx != 0:
            app.idx = 0
            self.render_current_row()

    def action_last_row(self) -> None:
        app = self.app_state
        last = len(app.rows) - 1
        if app.idx != last:
            app.idx = last
            self.render_current_row()

    def action_open_picker(self) -> None:
        app = self.app_state
        files = list_parquet_files(app.dataset_dir)
        if not files:
            return

        def on_picker_dismiss(path: Path | None) -> None:
            if path is not None:
                app.load_file(path)
                self.render_current_row()

        self.app.push_screen(
            PickerScreen(files, app.dataset_dir, prefer_latest=app.latest),
            callback=on_picker_dismiss,
        )

    def action_quit_app(self) -> None:
        self.app.exit()


class TrajectoryTuiApp(App):
    """Thin coordinator — holds shared state, pushes screens."""

    CSS = APP_CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        *,
        dataset_dir: Path,
        latest: bool,
        initial_path: Path | None,
        start_with_picker: bool,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.latest = latest
        self.path = initial_path
        self.rows: list[dict[str, Any]] = []
        self.idx = 0
        self.start_with_picker = start_with_picker

    def on_mount(self) -> None:
        if self.start_with_picker:
            files = list_parquet_files(self.dataset_dir)
            if not files:
                self.exit()
                return

            def on_initial_pick(path: Path | None) -> None:
                if path is None:
                    self.exit()
                    return
                self.load_file(path)
                self.push_screen(ViewerScreen())

            self.push_screen(
                PickerScreen(files, self.dataset_dir, prefer_latest=self.latest),
                callback=on_initial_pick,
            )
        else:
            if self.path is None:
                self.exit()
                return
            self.load_file(self.path)
            self.push_screen(ViewerScreen())

    def load_file(self, path: Path) -> None:
        self.path = path
        self.rows = load_rows(path)
        if not self.rows:
            raise ValueError(f"No rows found in {path}")
        self.idx = 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run(
    path: str | None = None,
    latest: bool = True,
    dataset_dir: str = "dataset_files",
    select: bool = True,
) -> None:
    dataset_dir_path = Path(dataset_dir)
    initial_path: Path | None = None
    start_with_picker = False

    if path is not None:
        initial_path = Path(path)
    elif select:
        start_with_picker = True
    elif latest:
        initial_path = resolve_default_path(dataset_dir_path)
    else:
        raise ValueError("When using latest=False, you must provide path.")

    app = TrajectoryTuiApp(
        dataset_dir=dataset_dir_path,
        latest=latest,
        initial_path=initial_path,
        start_with_picker=start_with_picker,
    )
    app.run()


if __name__ == "__main__":
    fire.Fire(run)
