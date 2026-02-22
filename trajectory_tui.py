#!/usr/bin/env python3
"""TUI for reviewing generated trajectory parquet files using Textual."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import fire
import pandas as pd
from rich.text import Text
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
            lines.append(f"{pad}{preview_value(block, max_len=300)}")
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
                lines.append(f"{pad}{line}")
        elif block_type == "tool_result":
            lines.append(f"{pad}[Tool Result]")
            for line in render_fields(block.get("content", "")):
                lines.append(f"{pad}{line}")
        else:
            lines.append(f"{pad}[{block_type or 'Content'}]")
            for line in render_fields(block):
                lines.append(f"{pad}{line}")


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
                lines.append(f"{pad}{line}")
        else:
            lines.append(f"{pad}[Tool Call]")
            lines.append(f"{pad}{preview_value(call, max_len=300)}")


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
        return [
            f"{key}: {preview_value(value, max_len=200)}" for key, value in obj.items()
        ]
    if isinstance(obj, list):
        return [preview_value(item, max_len=200) for item in obj]
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


def preview_value(value: Any, max_len: int = 100) -> str:
    value = normalize_value(value)
    if isinstance(value, str):
        return clean_message(value, truncate_length=max_len)
    if isinstance(value, (dict, list, bool, int, float)):
        return clean_message(value, truncate_length=max_len)

    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


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
            lines.append(f"{key}: {preview_value(value, max_len=120)}")
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
        lines.append(f"{key}: {preview_value(value, max_len=180)}")
    return lines


def render_row_text(
    path: Path,
    row: dict[str, Any],
    idx: int,
    total: int,
) -> tuple[str, str]:
    row_type = detect_row_type(row)
    header = f"Trajectory Viewer - {path.name} | row {idx + 1}/{total} | type: {row_type}"

    if row_type == "roleplay":
        body_lines = render_roleplay_lines(row)
    elif row_type == "conversation":
        body_lines = render_conversation_lines(row)
    else:
        body_lines = render_generic_lines(row)

    body_text = "\n".join(body_lines) if body_lines else ""
    return header, body_text


ROLE_LINE_STYLES: dict[str, str] = {
    "DUNGEON_MASTER": "bold yellow",
    "PLAYER": "bold green",
    "ASSISTANT": "bold cyan",
    "USER": "bold blue",
}

BLOCK_LINE_STYLES: dict[str, str] = {
    "[Text]": "bold cyan",
    "[Thinking]": "bold magenta",
    "[Tool Call": "bold yellow",
    "[Tool Result]": "bold green",
}

ROLE_LINE_RE = re.compile(r"^(\d+\.\s+)\[([A-Z_]+)\](.*)$")


def style_viewer_line(line: str) -> Text:
    indent_len = len(line) - len(line.lstrip(" "))
    indent = line[:indent_len]
    stripped = line[indent_len:]
    styled = Text(indent)

    if stripped.startswith("== ") and stripped.endswith(" =="):
        styled.append(stripped, style="bold magenta")
        return styled

    for prefix, style in BLOCK_LINE_STYLES.items():
        if stripped.startswith(prefix):
            styled.append(prefix, style=style)
            styled.append(stripped[len(prefix) :])
            return styled

    role_match = ROLE_LINE_RE.match(stripped)
    if role_match:
        item_prefix, role, suffix = role_match.groups()
        styled.append(item_prefix)
        styled.append(f"[{role}]", style=ROLE_LINE_STYLES.get(role, "bold"))
        styled.append(suffix)
        return styled

    if ":" in stripped:
        key, rest = stripped.split(":", 1)
        if key and " " not in key and not key.startswith("{"):
            styled.append(key, style="cyan")
            styled.append(f":{rest}")
            return styled

    styled.append(stripped)
    return styled


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


class TrajectoryTuiApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
    }

    #body {
        height: 1fr;
        width: 100%;
        border: round $accent;
        padding: 0 1;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: $boost;
    }

    #help {
        height: 2;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
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
        self.viewer_styled_lines: list[Text] = []
        self.viewer_scroll_offset = 0
        self.mode = "picker" if start_with_picker else "viewer"
        self.picker_files: list[Path] = []
        self.picker_line_payloads: list[str] = []
        self.picker_idx = 0
        self.picker_scroll_offset = 0
        self._last_body_cache_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield Static("", id="body")
        yield Static("", id="status")
        yield Static("", id="help")

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
            self.call_after_refresh(self.render_viewer_viewport)
        else:
            self.call_after_refresh(self.render_picker_viewport)

    def on_key(self, event: Key) -> None:
        if event.key == "ctrl+c":
            event.stop()
            self.exit()
            return

        if self.mode == "picker":
            self.handle_picker_key(event)
            return
        self.handle_viewer_key(event)

    def handle_picker_key(self, event: Key) -> None:
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

    def handle_viewer_key(self, event: Key) -> None:
        key = event.key
        ch = event.character or ""

        if ch == "j" or key == "down":
            event.stop()
            self.scroll_half_page(1)
            return
        if ch == "k" or key == "up":
            event.stop()
            self.scroll_half_page(-1)
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

    def scroll_half_page(self, direction: int) -> None:
        visible = self.body_height()
        half_page = max(1, visible // 2)
        max_scroll = max(0, len(self.viewer_lines) - visible)
        target = min(
            max(self.viewer_scroll_offset + direction * half_page, 0),
            max_scroll,
        )
        if target == self.viewer_scroll_offset:
            return
        self.viewer_scroll_offset = target
        self.render_viewer_viewport()

    def refresh_viewer(self, reset_scroll: bool = False) -> None:
        if self.path is None or not self.rows:
            return

        row = {k: normalize_value(v) for k, v in self.rows[self.idx].items()}
        _, body = render_row_text(
            self.path,
            row,
            self.idx,
            len(self.rows),
        )
        self.viewer_lines = body.splitlines() if body else []
        self.viewer_styled_lines = [style_viewer_line(line) for line in self.viewer_lines]

        row_type = detect_row_type(row)
        header_text = Text()
        header_text.append("Trajectory Viewer", style="bold")
        header_text.append(" - ")
        header_text.append(self.path.name, style="cyan")
        header_text.append(" | ")
        header_text.append(f"row {self.idx + 1}/{len(self.rows)}", style="green")
        header_text.append(" | ")
        header_text.append(f"type: {row_type}", style="magenta")
        self.query_one("#header", Static).update(header_text)

        help_text = Text()
        help_text.append("Keys: ", style="dim")
        help_text.append("j", style="bold")
        help_text.append("/Down half-page down, ")
        help_text.append("k", style="bold")
        help_text.append("/Up half-page up, ")
        help_text.append("n", style="bold")
        help_text.append("/Right/Enter next trajectory, ")
        help_text.append("N", style="bold")
        help_text.append("/Left previous trajectory, ")
        help_text.append("g", style="bold")
        help_text.append(" first, ")
        help_text.append("G", style="bold")
        help_text.append(" last, ")
        help_text.append("o", style="bold")
        help_text.append(" open file, ")
        help_text.append("q", style="bold")
        help_text.append(" quit")
        self.query_one("#help", Static).update(help_text)

        if reset_scroll:
            self.viewer_scroll_offset = 0
        self.render_viewer_viewport()

    def enter_picker(self, prefer_latest: bool) -> None:
        self.mode = "picker"
        self.picker_files = list_parquet_files(self.dataset_dir)
        if not self.picker_files:
            raise FileNotFoundError(f"No parquet files found in {self.dataset_dir}")
        self.picker_idx = len(self.picker_files) - 1 if prefer_latest else 0
        self.picker_scroll_offset = 0
        self.picker_line_payloads = []
        for file_path in self.picker_files:
            stat = file_path.stat()
            modified = str(pd.Timestamp.fromtimestamp(stat.st_mtime))
            self.picker_line_payloads.append(
                f"{file_path.name} | modified: {modified} | size: {stat.st_size:,} B"
            )
        self.refresh_picker_view()

    def refresh_picker_view(self) -> None:
        self.query_one("#header", Static).update(
            Text.assemble(
                ("Trajectory Viewer", "bold"),
                (" - Select parquet file ", ""),
                (f"({self.picker_idx + 1}/{len(self.picker_files)})", "green"),
            )
        )
        picker_help = Text()
        picker_help.append("Keys: ", style="dim")
        picker_help.append("j", style="bold")
        picker_help.append("/Down next, ")
        picker_help.append("k", style="bold")
        picker_help.append("/Up previous, ")
        picker_help.append("g", style="bold")
        picker_help.append(" first, ")
        picker_help.append("G", style="bold")
        picker_help.append(" last, ")
        picker_help.append("Enter", style="bold")
        picker_help.append(" select, ")
        picker_help.append("q", style="bold")
        picker_help.append(" cancel")
        self.query_one("#help", Static).update(
            picker_help
        )
        self.render_picker_viewport()

    def body_height(self) -> int:
        return max(1, self.query_one("#body", Static).size.height)

    def set_body_renderable(self, renderable: str | Text, cache_key: str) -> None:
        if cache_key == self._last_body_cache_key:
            return
        self.query_one("#body", Static).update(renderable)
        self._last_body_cache_key = cache_key

    def render_viewer_viewport(self) -> None:
        visible = self.body_height()
        total = len(self.viewer_lines)
        max_scroll = max(0, total - visible)
        self.viewer_scroll_offset = min(max(self.viewer_scroll_offset, 0), max_scroll)
        start = self.viewer_scroll_offset
        end = min(start + visible, total)
        viewport_text = Text()
        for i in range(start, end):
            viewport_text.append_text(self.viewer_styled_lines[i])
            if i < end - 1:
                viewport_text.append("\n")
        self.set_body_renderable(
            viewport_text,
            cache_key=f"viewer:{self.idx}:{start}:{end}:{total}",
        )

        if total == 0:
            display_start = 0
            display_end = 0
        else:
            display_start = start + 1
            display_end = end
        self.query_one("#status", Static).update(
            Text.assemble(
                ("Scroll: ", "dim"),
                (f"{display_start}-{display_end}/{total}", "bold"),
            )
        )

    def render_picker_viewport(self) -> None:
        visible = self.body_height()
        total = len(self.picker_line_payloads)
        if total == 0:
            self.set_body_renderable("", cache_key="picker:empty")
            self.query_one("#status", Static).update(
                Text.assemble(("dataset_dir: ", "dim"), (str(self.dataset_dir), "cyan"))
            )
            return

        max_scroll = max(0, total - visible)
        self.picker_scroll_offset = min(
            max(self.picker_scroll_offset, 0),
            max_scroll,
        )
        if self.picker_idx < self.picker_scroll_offset:
            self.picker_scroll_offset = self.picker_idx
        elif self.picker_idx >= self.picker_scroll_offset + visible:
            self.picker_scroll_offset = self.picker_idx - visible + 1

        start = self.picker_scroll_offset
        end = min(start + visible, total)
        viewport: list[str] = []
        for i in range(start, end):
            marker = ">" if i == self.picker_idx else " "
            viewport.append(f"{marker} {self.picker_line_payloads[i]}")
        self.set_body_renderable(
            "\n".join(viewport),
            cache_key=f"picker:{start}:{end}:{self.picker_idx}:{total}",
        )
        self.query_one("#status", Static).update(
            Text.assemble(
                ("dataset_dir: ", "dim"),
                (str(self.dataset_dir), "cyan"),
                (" | showing ", "dim"),
                (f"{start + 1}-{end}/{total}", "bold"),
            )
        )


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
