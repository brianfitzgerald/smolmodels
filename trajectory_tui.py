#!/usr/bin/env python3
"""TUI for reviewing generated trajectory parquet files using Textual."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import fire
import pandas as pd
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.events import Key, Resize
from textual.widgets import Static

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


def append_message(lines: list[str], message: Message | Any, index: int, indent: int = 0) -> None:
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
# Visual styling layer (Rich markup for Textual display)
# ---------------------------------------------------------------------------

ROLE_COLORS: dict[str, str] = {
    "DUNGEON_MASTER": "#e3b341",
    "PLAYER": "#7ee787",
    "ASSISTANT": "#79c0ff",
    "USER": "#d2a8ff",
    "SYSTEM": "#f0883e",
}

BLOCK_COLORS: dict[str, str] = {
    "Text": "#79c0ff",
    "Thinking": "#bc8cff",
    "Content": "#8b949e",
    "Tool Call": "#ffa657",
    "Tool Result": "#3fb950",
    "Tool Result Error": "#f85149",
}

ROLE_LINE_RE = re.compile(r"^(\d+\.\s+)\[([A-Z_]+)\](.*)$")
SECTION_LINE_RE = re.compile(r"^==\s*(.+?)\s*==$")
TOOL_CALL_LINE_RE = re.compile(r"^\[Tool Call:\s*(.+?)\]$")
TOOL_RESULT_LINE_RE = re.compile(r"^\[Tool Result(?::\s*(.+?))?(?:\s+ERROR)?\]$")
KEY_VALUE_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+):(.*)$")

SECTION_RULE = "\u2500" * 44  # ────────────


def style_viewer_line(line: str) -> str:
    """Apply Rich markup to a single plain-text viewer line."""
    indent_len = len(line) - len(line.lstrip(" "))
    indent = line[:indent_len]
    stripped = line[indent_len:]
    pad = escape(indent)

    if not stripped:
        return ""

    # ── SECTION ────────────────────────────────
    m = SECTION_LINE_RE.match(stripped)
    if m:
        title = m.group(1).upper()
        return (
            f"{pad}[#58a6ff]\u2500\u2500[/] "
            f"[bold #e6edf3]{escape(title)}[/] "
            f"[#58a6ff]{SECTION_RULE}[/]"
        )

    # ▸ Tool Call: name
    m = TOOL_CALL_LINE_RE.match(stripped)
    if m:
        name = m.group(1)
        return (
            f"{pad}[{BLOCK_COLORS['Tool Call']}]\u25b8 Tool Call[/]"
            f"  [{BLOCK_COLORS['Tool Call']}]{escape(name)}[/]"
        )

    # ▸ Tool Result / Tool Result ERROR
    m = TOOL_RESULT_LINE_RE.match(stripped)
    if m:
        ref = m.group(1)
        is_err = stripped.endswith(" ERROR]")
        color = BLOCK_COLORS["Tool Result Error"] if is_err else BLOCK_COLORS["Tool Result"]
        label = "Tool Result ERROR" if is_err else "Tool Result"
        suffix = f"  [#c9d1d9]{escape(ref)}[/]" if ref else ""
        return f"{pad}[{color}]\u25b8 {label}[/]{suffix}"

    # ▸ Text / Thinking / Content
    for name in ("Text", "Thinking", "Content"):
        if stripped == f"[{name}]":
            return f"{pad}[{BLOCK_COLORS[name]}]\u25b8 {name}[/]"

    # 1. [ROLE] — action or message role badge
    m = ROLE_LINE_RE.match(stripped)
    if m:
        num, role, suffix = m.groups()
        color = ROLE_COLORS.get(role, "#c9d1d9")
        return (
            f"{pad}[#484f58]{escape(num)}[/]"
            f"[bold #0d1117 on {color}] {escape(role)} [/]"
            f"[#c9d1d9]{escape(suffix)}[/]"
        )

    # key: value
    m = KEY_VALUE_LINE_RE.match(stripped)
    if m:
        key, val = m.groups()
        val = val.lstrip()
        if indent_len >= 4:
            return (
                f"{pad}[#484f58]\u00b7[/] "
                f"[#8b949e]{escape(key)}[/]"
                f"[#6e7681]:[/] "
                f"[#c9d1d9]{escape(val)}[/]"
            )
        return (
            f"{pad}[#8b949e]{escape(key)}[/]"
            f"[#6e7681]:[/] "
            f"[#e6edf3]{escape(val)}[/]"
        )

    # JSON-ish or plain text
    if stripped.startswith("{") or stripped.startswith("["):
        return f"{pad}[#6e7681]{escape(stripped)}[/]"
    return f"{pad}[#c9d1d9]{escape(stripped)}[/]"


def style_viewer_lines(lines: list[str]) -> list[str]:
    """Style all lines, rendering thinking-block body text in italic dim."""
    styled: list[str] = []
    in_thinking = False
    thinking_indent = 0

    for line in lines:
        indent_len = len(line) - len(line.lstrip(" "))
        stripped = line[indent_len:]

        if not stripped:
            styled.append("")
            continue

        if stripped == "[Thinking]":
            in_thinking = True
            thinking_indent = indent_len
            styled.append(style_viewer_line(line))
            continue

        if stripped.startswith("[") and stripped.endswith("]") and indent_len <= thinking_indent:
            in_thinking = False

        if in_thinking and indent_len >= thinking_indent:
            pad = escape(line[:indent_len])
            styled.append(f"{pad}[italic #6e7681]{escape(stripped)}[/]")
            continue

        styled.append(style_viewer_line(line))

    return styled


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
# Textual app
# ---------------------------------------------------------------------------

# Separator used in header/status bars
_SEP = "  [#30363d]\u2502[/]  "


def _key_hint(key: str, desc: str) -> str:
    return f"[bold #c9d1d9]{key}[/] [#6e7681]{desc}[/]"


class TrajectoryTuiApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: #0d1117;
    }

    #header {
        height: 1;
        padding: 0 1;
        background: #161b22;
        color: #e6edf3;
        text-style: bold;
    }

    #body {
        height: 1fr;
        width: 100%;
        border: round #30363d;
        background: #0d1117;
        padding: 0 1;
        color: #c9d1d9;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #161b22;
        color: #8b949e;
    }

    #help {
        height: auto;
        padding: 0 1;
        color: #6e7681;
        background: #0d1117;
    }
    """

    mode: Literal["viewer", "picker"]

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
        self.viewer_lines: list[str] = []
        self.viewer_styled_lines: list[str] = []
        self.viewer_scroll_offset = 0
        self.mode = "picker" if start_with_picker else "viewer"
        self.picker_files: list[Path] = []
        self.picker_line_payloads: list[str] = []
        self.picker_idx = 0
        self.picker_scroll_offset = 0
        self._last_body_cache_key: str | None = None

    # -- compose / lifecycle ------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="header", markup=True)
        yield Static("", id="body", markup=True)
        yield Static("", id="status", markup=True)
        yield Static("", id="help", markup=True)

    def on_mount(self) -> None:
        if self.mode == "picker":
            self.enter_picker(prefer_latest=self.latest)
            return
        if self.path is None:
            self.exit()
            return
        self.open_path(self.path)

    def on_resize(self, _event: Resize) -> None:
        if self.mode == "viewer":
            self.call_after_refresh(self._render_viewer_viewport)
        else:
            self.call_after_refresh(self._render_picker_viewport)

    # -- key handling -------------------------------------------------------

    def on_key(self, event: Key) -> None:
        if event.key == "ctrl+c":
            event.stop()
            self.exit()
            return

        if self.mode == "picker":
            self._handle_picker_key(event)
            return
        self._handle_viewer_key(event)

    def _handle_picker_key(self, event: Key) -> None:
        key = event.key
        ch = event.character or ""

        if ch in ("j", "n") or key == "down":
            event.stop()
            self.picker_idx = min(self.picker_idx + 1, len(self.picker_files) - 1)
            self.refresh_picker_view()
            return
        if ch in ("k", "p") or key == "up":
            event.stop()
            self.picker_idx = max(self.picker_idx - 1, 0)
            self.refresh_picker_view()
            return
        if ch == "g":
            event.stop()
            self.picker_idx = 0
            self.refresh_picker_view()
            return
        if ch == "G":
            event.stop()
            self.picker_idx = max(0, len(self.picker_files) - 1)
            self.refresh_picker_view()
            return
        if key == "enter":
            event.stop()
            if not self.picker_files:
                return
            selected = self.picker_files[self.picker_idx]
            self.mode = "viewer"
            self.open_path(selected)
            return
        if ch == "q" or key == "escape":
            event.stop()
            if self.path is None:
                self.exit()
                return
            self.mode = "viewer"
            self.refresh_viewer()

    def _handle_viewer_key(self, event: Key) -> None:
        key = event.key
        ch = event.character or ""

        if ch == "j" or key == "down":
            event.stop()
            self._scroll(1)
            return
        if ch == "k" or key == "up":
            event.stop()
            self._scroll(-1)
            return
        if key in ("enter", "right", "space") or ch in ("n", "l"):
            event.stop()
            self.move_row(1)
            return
        if key == "left" or ch in ("N", "p", "h"):
            event.stop()
            self.move_row(-1)
            return
        if ch == "g":
            event.stop()
            self.jump_to_row(0)
            return
        if ch == "G":
            event.stop()
            self.jump_to_row(len(self.rows) - 1)
            return
        if ch == "o":
            event.stop()
            self.enter_picker(prefer_latest=self.latest)
            return
        if ch == "q":
            event.stop()
            self.exit()

    # -- navigation ---------------------------------------------------------

    def open_path(self, path: Path) -> None:
        self.path = path
        self.rows = load_rows(path)
        if not self.rows:
            raise ValueError(f"No rows found in {path}")
        self.idx = 0
        self.refresh_viewer(reset_scroll=True)

    def move_row(self, delta: int) -> None:
        if not self.rows:
            return
        next_idx = min(max(self.idx + delta, 0), len(self.rows) - 1)
        if next_idx == self.idx:
            return
        self.idx = next_idx
        self.refresh_viewer(reset_scroll=True)

    def jump_to_row(self, target_idx: int) -> None:
        if not self.rows:
            return
        bounded = min(max(target_idx, 0), len(self.rows) - 1)
        if bounded == self.idx:
            return
        self.idx = bounded
        self.refresh_viewer(reset_scroll=True)

    def _scroll(self, direction: int) -> None:
        visible = self._body_height()
        step = max(1, visible // 4)
        limit = max(0, len(self.viewer_lines) - visible)
        target = min(max(self.viewer_scroll_offset + direction * step, 0), limit)
        if target == self.viewer_scroll_offset:
            return
        self.viewer_scroll_offset = target
        self._render_viewer_viewport()

    # -- viewer -------------------------------------------------------------

    def refresh_viewer(self, reset_scroll: bool = False) -> None:
        if self.path is None or not self.rows:
            return

        row = {k: normalize_value(v) for k, v in self.rows[self.idx].items()}
        body = render_row_text(row)
        self.viewer_lines = body.splitlines() if body else []
        self.viewer_styled_lines = style_viewer_lines(self.viewer_lines)

        row_type = detect_row_type(row)
        self.query_one("#header", Static).update(self._viewer_header(row_type))
        self.query_one("#help", Static).update(self._viewer_help())

        if reset_scroll:
            self.viewer_scroll_offset = 0
        self._render_viewer_viewport()

    def _body_height(self) -> int:
        return max(1, self.query_one("#body", Static).size.height)

    def _set_body(self, markup: str, cache_key: str) -> None:
        if cache_key == self._last_body_cache_key:
            return
        self.query_one("#body", Static).update(markup)
        self._last_body_cache_key = cache_key

    def _render_viewer_viewport(self) -> None:
        visible = self._body_height()
        total = len(self.viewer_lines)
        max_off = max(0, total - visible)
        self.viewer_scroll_offset = min(max(self.viewer_scroll_offset, 0), max_off)
        start = self.viewer_scroll_offset
        end = min(start + visible, total)

        self._set_body(
            "\n".join(self.viewer_styled_lines[start:end]),
            cache_key=f"v:{self.idx}:{start}:{end}:{total}",
        )

        if total == 0:
            d_start, d_end = 0, 0
        else:
            d_start, d_end = start + 1, end

        self.query_one("#status", Static).update(
            f"[#6e7681]Lines[/] "
            f"[bold #c9d1d9]{d_start}\u2013{d_end}[/]"
            f"[#484f58]/[/]"
            f"[#8b949e]{total}[/]"
            f"{_SEP}"
            f"[#6e7681]Row[/] "
            f"[bold #c9d1d9]{self.idx + 1}[/]"
            f"[#484f58]/[/]"
            f"[#8b949e]{len(self.rows)}[/]"
        )

    def _viewer_header(self, row_type: str) -> str:
        name = self.path.name if self.path is not None else ""
        return (
            "[bold #e6edf3]Trajectory Viewer[/]"
            f"{_SEP}"
            f"[#58a6ff]{escape(name)}[/]"
            f"{_SEP}"
            f"[bold #3fb950]{self.idx + 1}/{len(self.rows)}[/]"
            f"{_SEP}"
            f"[#bc8cff]{escape(row_type)}[/]"
        )

    def _viewer_help(self) -> str:
        w = max(40, self.size.width)
        if w < 110:
            return (
                _key_hint("j/k", "scroll") + "  "
                + _key_hint("n/N", "next/prev") + "  "
                + _key_hint("g/G", "first/last") + "  "
                + _key_hint("o", "file") + "  "
                + _key_hint("q", "quit")
            )
        return (
            _key_hint("j/\u2193", "scroll down") + "   "
            + _key_hint("k/\u2191", "scroll up") + "   "
            + _key_hint("n/\u2192/Enter", "next") + "   "
            + _key_hint("N/\u2190", "prev") + "   "
            + _key_hint("g", "first") + "   "
            + _key_hint("G", "last") + "   "
            + _key_hint("o", "open file") + "   "
            + _key_hint("q", "quit")
        )

    # -- picker -------------------------------------------------------------

    def enter_picker(self, prefer_latest: bool) -> None:
        self.mode = "picker"
        self.picker_files = list_parquet_files(self.dataset_dir)
        if not self.picker_files:
            raise FileNotFoundError(f"No parquet files found in {self.dataset_dir}")
        self.picker_idx = len(self.picker_files) - 1 if prefer_latest else 0
        self.picker_scroll_offset = 0
        self.picker_line_payloads = []
        for fp in self.picker_files:
            stat = fp.stat()
            modified = str(pd.Timestamp.fromtimestamp(stat.st_mtime))
            self.picker_line_payloads.append(
                f"{fp.name}  \u2502  {modified}  \u2502  {stat.st_size:,} B"
            )
        self.refresh_picker_view()

    def refresh_picker_view(self) -> None:
        self.query_one("#header", Static).update(self._picker_header())
        self.query_one("#help", Static).update(self._picker_help())
        self._render_picker_viewport()

    def _render_picker_viewport(self) -> None:
        visible = self._body_height()
        total = len(self.picker_line_payloads)
        if total == 0:
            self._set_body("", cache_key="picker:empty")
            self.query_one("#status", Static).update(
                f"[#6e7681]Dir[/] [#58a6ff]{escape(str(self.dataset_dir))}[/]"
            )
            return

        max_off = max(0, total - visible)
        self.picker_scroll_offset = min(max(self.picker_scroll_offset, 0), max_off)
        if self.picker_idx < self.picker_scroll_offset:
            self.picker_scroll_offset = self.picker_idx
        elif self.picker_idx >= self.picker_scroll_offset + visible:
            self.picker_scroll_offset = self.picker_idx - visible + 1

        start = self.picker_scroll_offset
        end = min(start + visible, total)
        rows: list[str] = []
        for i in range(start, end):
            text = escape(self.picker_line_payloads[i])
            if i == self.picker_idx:
                rows.append(f"[bold #0d1117 on #58a6ff] \u25b8 {text} [/]")
            else:
                rows.append(f"  [#8b949e]{text}[/]")
        self._set_body(
            "\n".join(rows),
            cache_key=f"p:{start}:{end}:{self.picker_idx}:{total}",
        )
        self.query_one("#status", Static).update(
            f"[#6e7681]Dir[/] [#58a6ff]{escape(str(self.dataset_dir))}[/]"
            f"{_SEP}"
            f"[bold #c9d1d9]{start + 1}\u2013{end}[/]"
            f"[#484f58]/[/]"
            f"[#8b949e]{total}[/]"
        )

    def _picker_header(self) -> str:
        return (
            "[bold #e6edf3]Trajectory Viewer[/]"
            f"{_SEP}"
            "[#58a6ff]Select File[/]"
            f"{_SEP}"
            f"[bold #3fb950]{self.picker_idx + 1}/{len(self.picker_files)}[/]"
        )

    def _picker_help(self) -> str:
        w = max(40, self.size.width)
        if w < 100:
            return (
                _key_hint("j/k", "move") + "  "
                + _key_hint("g/G", "ends") + "  "
                + _key_hint("Enter", "open") + "  "
                + _key_hint("q", "cancel")
            )
        return (
            _key_hint("j/\u2193", "next") + "   "
            + _key_hint("k/\u2191", "prev") + "   "
            + _key_hint("g", "first") + "   "
            + _key_hint("G", "last") + "   "
            + _key_hint("Enter", "select") + "   "
            + _key_hint("q", "cancel")
        )


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
