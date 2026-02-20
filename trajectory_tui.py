#!/usr/bin/env python3
"""Minimal TUI for reviewing roleplaying trajectory parquet files."""

from __future__ import annotations

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


def indent_block(text: str, spaces: int = 2) -> str:
    if not text:
        return ""
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" if line else "" for line in text.splitlines())


def render_roleplay(row: dict[str, Any]) -> list[Any]:
    rows: list[Any] = []

    transcript = row.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(
            "This TUI requires the new roleplaying format with a non-empty `transcript` field."
        )

    info = Table.grid(padding=(0, 2))
    info.add_column(style="cyan")
    info.add_column()
    info.add_row("game_setting", str(row.get("game_setting", "")))
    info.add_row("player_character", str(row.get("player_character", "")))
    info.add_row("step_count", str(row.get("step_count", "")))
    rows.append(Panel(info, title="Episode"))

    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metrics:
        metrics_table = Table(title="Metrics", show_header=True, header_style="bold magenta")
        metrics_table.add_column("key")
        metrics_table.add_column("value")
        for key, value in metrics.items():
            metrics_table.add_row(str(key), str(value))
        rows.append(metrics_table)

    rows.append(Panel(Text(transcript), title="Transcript"))
    return rows


def render_row(
    console: Console,
    path: Path,
    row: dict[str, Any],
    idx: int,
    total: int,
    scroll_offset: int = 0,
) -> tuple[Group, int, int]:
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        f"[bold]Trajectory Viewer[/bold] - {path.name}",
        f"row {idx + 1}/{total}",
    )
    top_panel = Panel(header)

    panels = render_roleplay(row)

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
        while True:
            row = rows[idx]
            view, max_scroll, page_step = render_row(
                console,
                resolved_path,
                row,
                idx,
                len(rows),
                scroll_offset=scroll_offset,
            )
            live.update(view, refresh=True)

            key = read_key()
            if key == "":
                break
            if key in ("j", "\x1b[B"):
                scroll_offset = min(scroll_offset + page_step, max_scroll)
            elif key in ("k", "\x1b[A"):
                scroll_offset = max(scroll_offset - page_step, 0)
            elif key in ("\r", "\n", " ", "n", "l", "\x1b[C"):
                idx = min(idx + 1, len(rows) - 1)
                scroll_offset = 0
            elif key in ("N", "p", "h", "\x1b[D"):
                idx = max(idx - 1, 0)
                scroll_offset = 0
            elif key == "g":
                idx = 0
                scroll_offset = 0
            elif key == "G":
                idx = len(rows) - 1
                scroll_offset = 0
            elif key == "o":
                live.stop()
                selected = select_parquet_file(
                    console, dataset_dir_path, prefer_latest=latest
                )
                live.start(refresh=True)
                if selected is None:
                    continue
                resolved_path = selected
                rows = load_rows(resolved_path)
                if not rows:
                    raise ValueError(f"No rows found in {resolved_path}")
                idx = 0
                scroll_offset = 0
            elif key in ("q", "\x03"):
                break


if __name__ == "__main__":
    fire.Fire(run)
