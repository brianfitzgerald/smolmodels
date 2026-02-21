#!/usr/bin/env python3
"""Basic TUI for reviewing generated trajectory parquet files."""

from __future__ import annotations

import json
import sys
import termios
import tty
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


def format_message(message: Message | Any, index: int) -> str:
    message = normalize_value(message)
    if not isinstance(message, dict):
        return f"{index}. {message}"

    role = str(message.get("role", "unknown")).upper()
    parts = [f"{index}. [{role}]"]
    body = format_message_body(message)
    if body:
        parts.append(indent_block(body))
    return "\n".join(parts)


def indent_block(text: str, spaces: int = 2) -> str:
    if not text:
        return ""
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" if line else "" for line in text.splitlines())


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


def format_content_blocks(content: Any) -> str:
    blocks = parse_content_blocks(content)
    rendered: list[str] = []
    for block in blocks:
        block = normalize_value(block)
        if isinstance(block, str):
            rendered.append("[Text]")
            rendered.append(block)
            continue
        if not isinstance(block, dict):
            rendered.append("[Content]")
            rendered.append(preview_value(block, max_len=300))
            continue

        block_type = block.get("type", "")
        if block_type == "text":
            rendered.append("[Text]")
            rendered.append(str(block.get("text", "")))
        elif block_type == "thinking":
            rendered.append("[Thinking]")
            rendered.append(str(block.get("thinking", "")))
        elif block_type == "tool_use":
            tool_block: ToolUseBlock = block  # type: ignore[assignment]
            name = str(block.get("name", "unknown"))
            rendered.append(f"[Tool Call: {name}]")
            for line in render_arg_fields(tool_block.get("input", {})):
                rendered.append(line)
        elif block_type == "tool_result":
            result_block: ToolResultBlock = block  # type: ignore[assignment]
            rendered.append("[Tool Result]")
            for line in render_fields(result_block.get("content", "")):
                rendered.append(line)
        else:
            rendered.append(f"[{block_type or 'Content'}]")
            for line in render_fields(block):
                rendered.append(line)
    return "\n".join(rendered)


def format_message_body(message: Message | dict[str, Any]) -> str:
    parts: list[str] = []
    content = format_content_blocks(message.get("content", ""))
    if content:
        parts.append(content)

    tool_calls = format_tool_calls(message.get("tool_calls"))
    if tool_calls:
        parts.append(tool_calls)
    return "\n".join(parts)


def format_tool_calls(tool_calls: Any) -> str:
    normalized = normalize_tool_calls(tool_calls)
    if not normalized:
        return ""

    formatted: list[str] = []
    for call in normalized:
        if isinstance(call, dict):
            fn = call.get("function", {})
            if isinstance(fn, dict):
                name = fn.get("name", "unknown")
                args = fn.get("arguments", {})
            else:
                name = "unknown"
                args = {}
            formatted.append(f"[Tool Call: {name}]")
            for line in render_arg_fields(args):
                formatted.append(line)
        else:
            formatted.append("[Tool Call]")
            formatted.append(preview_value(call, max_len=300))
    return "\n".join(formatted)


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
                    conv.append(indent_block(format_message(message, j)))
                else:
                    conv.append(indent_block(f"{j}. {message}"))
                conv.append("\n")
        else:
            conv.append(indent_block("No messages"))
        conv.append("\n\n")

    panels.append(Panel(conv, title="Actions"))
    return panels


def render_conversation(row: dict[str, Any]) -> list[Any]:
    messages: Conversation = get_messages_from_row(row)
    text = Text()
    for i, message in enumerate(messages, start=1):
        text.append(format_message(message, i))
        text.append("\n\n")
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
    body_console = Console(width=body_width, record=True)
    for panel in panels:
        body_console.print(panel)
    body_lines = body_console.export_text().splitlines()

    viewport_height = max(console.size.height - 8, 8)
    page_step = max(1, viewport_height - 2)
    max_scroll = max(0, len(body_lines) - viewport_height)
    scroll_offset = min(max(scroll_offset, 0), max_scroll)
    visible_lines = body_lines[scroll_offset : scroll_offset + viewport_height]
    viewport_text = Text(
        "\n".join(visible_lines) if visible_lines else "",
        no_wrap=True,
        overflow="crop",
    )
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
