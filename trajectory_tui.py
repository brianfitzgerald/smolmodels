#!/usr/bin/env python3
"""Basic TUI for reviewing generated trajectory parquet files."""

from __future__ import annotations

import json
import sys
import termios
import tty
from io import StringIO
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.trajectory_formatting import (
    normalize_tool_calls,
    normalize_value,
    parse_content_blocks,
)
from synthetic_data.utils import (
    Conversation,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    clean_message,
    recursive_json_parse,
)

CONVERSATION_COLUMNS = ("conversation", "messages", "conversations", "trajectory")


# -- Styled rendering (appends directly to rich.text.Text objects) -----------


BLOCK_STYLES: dict[str, tuple[str, str]] = {
    "text": ("bold cyan", ""),
    "thinking": ("bold magenta", "dim italic"),
    "tool_use": ("bold yellow", "yellow"),
    "tool_result": ("bold green", "green"),
    "tool_result_error": ("bold red", "red"),
}


def _append_lines(target: Text, content: str, indent: int, style: str = "") -> None:
    pad = " " * indent
    for line in content.splitlines():
        target.append(f"{pad}{line}\n", style=style)


def append_content_blocks_to_text(
    target: Text, content: Any, indent: int = 0
) -> None:
    """Append formatted content blocks with rich styling to a Text object."""
    blocks = parse_content_blocks(content)
    pad = " " * indent
    for block in blocks:
        block = normalize_value(block)
        if isinstance(block, str):
            target.append(f"{pad}[Text]\n", style="bold cyan")
            _append_lines(target, block, indent)
            continue
        if not isinstance(block, dict):
            target.append(f"{pad}[Content]\n", style="bold cyan")
            target.append(f"{pad}{preview_value(block, max_len=300)}\n")
            continue

        block_type = block.get("type", "")
        if block_type == "text":
            target.append(f"{pad}[Text]\n", style="bold cyan")
            _append_lines(target, str(block.get("text", "")), indent)
        elif block_type == "thinking":
            target.append(f"{pad}[Thinking]\n", style="bold magenta")
            _append_lines(
                target, str(block.get("thinking", "")), indent, style="dim italic"
            )
        elif block_type == "tool_use":
            tool_block: ToolUseBlock = block  # type: ignore[assignment]
            name = str(block.get("name", "unknown"))
            target.append(f"{pad}[Tool Call: {name}]\n", style="bold yellow")
            for line in render_arg_fields(tool_block.get("input", {})):
                target.append(f"{pad}{line}\n", style="yellow")
        elif block_type == "tool_result":
            result_block: ToolResultBlock = block  # type: ignore[assignment]
            is_error = result_block.get("is_error", False)
            hdr_style, body_style = (
                BLOCK_STYLES["tool_result_error"]
                if is_error
                else BLOCK_STYLES["tool_result"]
            )
            target.append(f"{pad}[Tool Result]\n", style=hdr_style)
            for line in render_fields(result_block.get("content", "")):
                target.append(f"{pad}{line}\n", style=body_style)
        else:
            target.append(f"{pad}[{block_type or 'Content'}]\n", style="bold cyan")
            for line in render_fields(block):
                target.append(f"{pad}{line}\n")


def append_tool_calls_to_text(
    target: Text, tool_calls: Any, indent: int = 0
) -> None:
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
            target.append(f"{pad}[Tool Call: {name}]\n", style="bold yellow")
            for line in render_arg_fields(args):
                target.append(f"{pad}{line}\n", style="yellow")
        else:
            target.append(f"{pad}[Tool Call]\n", style="bold yellow")
            target.append(f"{pad}{preview_value(call, max_len=300)}\n")


def append_message_to_text(
    target: Text, message: Message | Any, index: int, indent: int = 0
) -> None:
    message = normalize_value(message)
    pad = " " * indent
    if not isinstance(message, dict):
        target.append(f"{pad}{index}. {message}\n")
        return

    role = str(message.get("role", "unknown")).upper()
    target.append(f"{pad}{index}. [{role}]\n", style="bold")
    body_indent = indent + 2
    content = message.get("content", "")
    if content:
        append_content_blocks_to_text(target, content, indent=body_indent)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        append_tool_calls_to_text(target, tool_calls, indent=body_indent)


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


ROLE_STYLES: dict[str, str] = {
    "dungeon_master": "bold yellow",
    "player": "bold green",
}


def render_roleplay(row: dict[str, Any]) -> list[Any]:
    panels: list[Any] = []

    # Episode info
    info = Table.grid(padding=(0, 2))
    info.add_column(style="cyan")
    info.add_column()
    info.add_row("game_setting", str(row.get("game_setting", "")))
    info.add_row("player_character", str(row.get("player_character", "")))
    info.add_row("step_count", str(row.get("step_count", "")))
    panels.append(Panel(info, title="Episode"))

    metadata = normalize_value(row.get("metadata"))
    if isinstance(metadata, dict) and metadata:
        md = Table(title="Metadata", show_header=True, header_style="bold magenta")
        md.add_column("key")
        md.add_column("value")
        for key, value in metadata.items():
            md.add_row(str(key), preview_value(value, max_len=120))
        panels.append(md)

    # Metrics
    metrics = normalize_value(row.get("metrics"))
    if isinstance(metrics, dict) and metrics:
        mt = Table.grid(padding=(0, 2))
        mt.add_column(style="cyan")
        mt.add_column()
        for key, value in metrics.items():
            mt.add_row(str(key), str(value))
        panels.append(Panel(mt, title="Metrics"))

    # Actions — each action has a game role and a list of API messages
    actions = normalize_value(row.get("actions")) or []
    if not isinstance(actions, list):
        actions = [actions]

    conv = Text()
    for i, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            conv.append(f"{i}. {action}\n\n")
            continue

        role = str(action.get("role", "unknown"))
        style = ROLE_STYLES.get(role, "bold")
        conv.append(f"{i}. [{role.upper()}]\n", style=style)

        action_messages = normalize_value(action.get("messages")) or []
        if not isinstance(action_messages, list):
            action_messages = [action_messages]

        if action_messages:
            for j, message in enumerate(action_messages, start=1):
                if isinstance(message, dict):
                    append_message_to_text(conv, message, j, indent=2)
                else:
                    conv.append(f"  {j}. {message}\n")
        else:
            conv.append("  No messages\n")
        conv.append("\n")

    panels.append(Panel(conv, title="Actions"))
    return panels


def render_conversation(row: dict[str, Any]) -> list[Any]:
    messages: Conversation = get_messages_from_row(row)
    text = Text()
    for i, message in enumerate(messages, start=1):
        append_message_to_text(text, message, i)
        text.append("\n")
    if not messages:
        text.append("No conversation messages detected.")
    return [Panel(text, title="Conversation")]


def render_generic(row: dict[str, Any]) -> list[Any]:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value")
    for key, value in row.items():
        table.add_row(str(key), preview_value(value, max_len=180))
    return [Panel(table, title="Row Fields")]


def render_row(
    console: Console,
    path: Path,
    row: dict[str, Any],
    idx: int,
    total: int,
    scroll_offset: int = 0,
) -> tuple[Group, int, int]:
    row_type = detect_row_type(row)
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        f"[bold]Trajectory Viewer[/bold] - {path.name}",
        f"row {idx + 1}/{total} | type: {row_type}",
    )
    top_panel = Panel(header)

    if row_type == "roleplay":
        panels = render_roleplay(row)
    elif row_type == "conversation":
        panels = render_conversation(row)
    else:
        panels = render_generic(row)

    body_width = max(20, console.size.width - 6)
    string_io = StringIO()
    body_console = Console(file=string_io, width=body_width, force_terminal=True)
    for panel in panels:
        body_console.print(panel)
    body_lines = string_io.getvalue().splitlines()

    viewport_height = max(console.size.height - 8, 8)
    page_step = max(1, viewport_height - 2)
    max_scroll = max(0, len(body_lines) - viewport_height)
    scroll_offset = min(max(scroll_offset, 0), max_scroll)
    visible_lines = body_lines[scroll_offset : scroll_offset + viewport_height]
    viewport_text = Text.from_ansi(
        "\n".join(visible_lines) if visible_lines else ""
    )
    viewport_text.no_wrap = True
    viewport_text.overflow = "crop"
    scroll_panel = Panel(viewport_text, title="Trajectory", padding=(0, 0))
    current_end = min(scroll_offset + viewport_height, len(body_lines))
    current_start = min(scroll_offset + 1, current_end) if body_lines else 0
    scroll_status = Text.from_markup(
        f"[dim]Scroll:[/dim] {current_start}-{current_end}/{len(body_lines)}"
    )
    key_help = Text.from_markup(
        "[dim]Keys:[/dim] [bold]j[/bold]/[bold]Down[/bold] page down, [bold]k[/bold]/[bold]Up[/bold] page up, [bold]n[/bold]/[bold]Right[/bold]/[bold]Enter[/bold] next trajectory, [bold]N[/bold]/[bold]Left[/bold] previous trajectory, [bold]g[/bold] first, [bold]G[/bold] last, [bold]o[/bold] open file, [bold]q[/bold] quit"
    )
    return Group(top_panel, scroll_panel, scroll_status, key_help), max_scroll, page_step


def read_key() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read(1)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return f"\x1b[{ch3}"
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def resolve_default_path(dataset_dir: Path) -> Path:
    parquet_files = list_parquet_files(dataset_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    return parquet_files[-1]


def list_parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)


def render_file_picker(
    console: Console, dataset_dir: Path, parquet_files: list[Path], idx: int
) -> None:
    console.clear()
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        "[bold]Trajectory Viewer[/bold] - Select parquet file",
        f"{idx + 1}/{len(parquet_files)}",
    )
    console.print(Panel(header))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("", no_wrap=True)
    table.add_column("file")
    table.add_column("modified", no_wrap=True)
    table.add_column("size", justify="right", no_wrap=True)
    for i, file_path in enumerate(parquet_files):
        marker = ">" if i == idx else " "
        stat = file_path.stat()
        table.add_row(
            marker,
            file_path.name,
            str(pd.Timestamp.fromtimestamp(stat.st_mtime)),
            f"{stat.st_size:,} B",
        )
    console.print(Panel(table, title=f"dataset_files: {dataset_dir}"))
    console.print(
        "[dim]Keys:[/dim] [bold]j[/bold]/[bold]Down[/bold] next, [bold]k[/bold]/[bold]Up[/bold] previous, [bold]g[/bold] first, [bold]G[/bold] last, [bold]Enter[/bold] select, [bold]q[/bold] cancel"
    )


def select_parquet_file(
    console: Console, dataset_dir: Path, prefer_latest: bool = True
) -> Path | None:
    parquet_files = list_parquet_files(dataset_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    idx = len(parquet_files) - 1 if prefer_latest else 0
    while True:
        render_file_picker(console, dataset_dir, parquet_files, idx)
        key = read_key()
        if key == "":
            return None
        if key in ("\r", "\n"):
            return parquet_files[idx]
        if key in ("j", "n", "\x1b[B"):
            idx = min(idx + 1, len(parquet_files) - 1)
        elif key in ("k", "p", "\x1b[A"):
            idx = max(idx - 1, 0)
        elif key == "g":
            idx = 0
        elif key == "G":
            idx = len(parquet_files) - 1
        elif key in ("q", "\x03"):
            return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def run(
    path: str | None = None,
    latest: bool = True,
    dataset_dir: str = "dataset_files",
    select: bool = True,
) -> None:
    dataset_dir_path = Path(dataset_dir)
    console = Console()

    if path is not None:
        resolved_path = Path(path)
    elif select:
        selected = select_parquet_file(
            console, dataset_dir_path, prefer_latest=latest
        )
        if selected is None:
            return
        resolved_path = selected
    elif latest:
        resolved_path = resolve_default_path(dataset_dir_path)
    else:
        raise ValueError("When using latest=False, you must provide path.")

    rows = load_rows(resolved_path)
    if not rows:
        raise ValueError(f"No rows found in {resolved_path}")

    idx = 0
    scroll_offset = 0
    with Live(console=console, screen=True, auto_refresh=False) as live:
        max_scroll = 0
        page_step = 1

        def refresh_view() -> None:
            nonlocal max_scroll, page_step
            row = {k: normalize_value(v) for k, v in rows[idx].items()}
            view, max_scroll, page_step = render_row(
                console,
                resolved_path,
                row,
                idx,
                len(rows),
                scroll_offset=scroll_offset,
            )
            live.update(view, refresh=True)

        refresh_view()
        while True:
            key = read_key()
            if key == "":
                break
            changed = False
            if key in ("j", "\x1b[B"):
                next_scroll = min(scroll_offset + page_step, max_scroll)
                changed = next_scroll != scroll_offset
                scroll_offset = next_scroll
            elif key in ("k", "\x1b[A"):
                next_scroll = max(scroll_offset - page_step, 0)
                changed = next_scroll != scroll_offset
                scroll_offset = next_scroll
            elif key in ("\r", "\n", " ", "n", "l", "\x1b[C"):
                next_idx = min(idx + 1, len(rows) - 1)
                changed = next_idx != idx or scroll_offset != 0
                idx = next_idx
                scroll_offset = 0
            elif key in ("N", "p", "h", "\x1b[D"):
                next_idx = max(idx - 1, 0)
                changed = next_idx != idx or scroll_offset != 0
                idx = next_idx
                scroll_offset = 0
            elif key == "g":
                changed = idx != 0 or scroll_offset != 0
                idx = 0
                scroll_offset = 0
            elif key == "G":
                last_idx = len(rows) - 1
                changed = idx != last_idx or scroll_offset != 0
                idx = last_idx
                scroll_offset = 0
            elif key == "o":
                live.stop()
                selected = select_parquet_file(
                    console, dataset_dir_path, prefer_latest=latest
                )
                live.start(refresh=True)
                if selected is None:
                    refresh_view()
                    continue
                resolved_path = selected
                rows = load_rows(resolved_path)
                if not rows:
                    raise ValueError(f"No rows found in {resolved_path}")
                idx = 0
                scroll_offset = 0
                changed = True
            elif key in ("q", "\x03"):
                break

            if changed:
                refresh_view()


if __name__ == "__main__":
    fire.Fire(run)
