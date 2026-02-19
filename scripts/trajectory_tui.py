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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from scripts.trajectory_formatting import (
        extract_text_from_content,
        normalize_tool_calls,
        normalize_value,
    )
except ModuleNotFoundError:
    from trajectory_formatting import (  # type: ignore[no-redef]
        extract_text_from_content,
        normalize_tool_calls,
        normalize_value,
    )

CONVERSATION_COLUMNS = ("conversation", "messages", "conversations", "trajectory")


def format_message(message: Any, index: int) -> str:
    message = normalize_value(message)
    if not isinstance(message, dict):
        return f"{index}. {message}"

    role = str(message.get("role", "unknown")).upper()
    content = extract_text_from_content(message.get("content", ""))
    tool_calls = format_tool_calls(message.get("tool_calls"))

    parts = [f"{index}. [{role}]"]
    if content:
        parts.append(content)
    if tool_calls:
        parts.append(tool_calls)
    return "\n".join(parts)


def format_tool_calls(tool_calls: Any) -> str:
    normalized = normalize_tool_calls(tool_calls)
    if not normalized:
        return ""

    formatted: list[str] = ["tool_calls:"]
    for call in normalized:
        formatted.append(json.dumps(call, indent=2, ensure_ascii=True))
    return "\n".join(formatted)


def get_messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
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
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def render_roleplay(row: dict[str, Any]) -> list[Any]:
    rows: list[Any] = []
    info = Table.grid(padding=(0, 2))
    info.add_column(style="cyan")
    info.add_column()
    info.add_row("game_setting", str(row.get("game_setting", "")))
    info.add_row("player_character", str(row.get("player_character", "")))
    info.add_row("step_count", str(row.get("step_count", "")))
    rows.append(Panel(info, title="Episode"))

    metadata = normalize_value(row.get("metadata"))
    if isinstance(metadata, dict) and metadata:
        md = Table(title="Metadata", show_header=True, header_style="bold magenta")
        md.add_column("key")
        md.add_column("value")
        for key, value in metadata.items():
            md.add_row(str(key), preview_value(value, max_len=120))
        rows.append(md)

    actions = normalize_value(row.get("actions")) or []
    if not isinstance(actions, list):
        actions = [actions]
    conv = Text()
    for i, action in enumerate(actions, start=1):
        role = "unknown"
        message = action
        if isinstance(action, dict) and isinstance(action.get("message"), dict):
            role = str(action.get("role", role))
            message = action["message"]
        elif isinstance(action, dict):
            role = str(action.get("role", role))
        label = role.upper()
        conv.append(f"{i}. [{label}]\n", style="bold yellow")
        if isinstance(message, dict):
            conv.append(extract_text_from_content(message.get("content", "")))
            tool_calls = format_tool_calls(message.get("tool_calls"))
            if tool_calls:
                conv.append("\n")
                conv.append(tool_calls)
        else:
            conv.append(str(message))
        conv.append("\n\n")
    rows.append(Panel(conv, title="Actions"))
    return rows


def render_conversation(row: dict[str, Any]) -> list[Any]:
    messages = get_messages_from_row(row)
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


def render_row(console: Console, path: Path, row: dict[str, Any], idx: int, total: int) -> None:
    console.clear()
    row_type = detect_row_type(row)
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        f"[bold]Trajectory Viewer[/bold] - {path.name}",
        f"row {idx + 1}/{total} | type: {row_type}",
    )
    console.print(Panel(header))

    if row_type == "roleplay":
        panels = render_roleplay(row)
    elif row_type == "conversation":
        panels = render_conversation(row)
    else:
        panels = render_generic(row)

    for panel in panels:
        console.print(panel)
    console.print(
        "[dim]Keys:[/dim] [bold]n[/bold]/[bold]Right[/bold]/[bold]Enter[/bold] next, [bold]p[/bold]/[bold]Left[/bold] previous, [bold]g[/bold] first, [bold]G[/bold] last, [bold]q[/bold] quit"
    )


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
    parquet_files = sorted(dataset_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    return parquet_files[-1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def run(path: str | None = None, latest: bool = True, dataset_dir: str = "dataset_files") -> None:
    dataset_dir_path = Path(dataset_dir)
    if latest:
        resolved_path = resolve_default_path(dataset_dir_path)
    elif path is not None:
        resolved_path = Path(path)
    else:
        raise ValueError("When using latest=False, you must provide path.")

    rows = load_rows(resolved_path)
    if not rows:
        raise ValueError(f"No rows found in {resolved_path}")

    console = Console()
    idx = 0
    while True:
        row = {k: normalize_value(v) for k, v in rows[idx].items()}
        render_row(console, resolved_path, row, idx, len(rows))

        key = read_key()
        if key == "":
            break
        if key in ("\r", "\n", " ", "n", "l", "\x1b[C"):
            idx = min(idx + 1, len(rows) - 1)
        elif key in ("p", "h", "\x1b[D"):
            idx = max(idx - 1, 0)
        elif key == "g":
            idx = 0
        elif key == "G":
            idx = len(rows) - 1
        elif key in ("q", "\x03"):
            break


if __name__ == "__main__":
    fire.Fire(run)
