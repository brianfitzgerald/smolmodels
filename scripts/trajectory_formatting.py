"""Shared parsing/formatting helpers for trajectory viewers."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np


def normalize_value(value: Any) -> Any:
    """Normalize values loaded from parquet (ndarrays, stringified literals)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value.__class__.__name__ == "Series" and hasattr(value, "to_list"):
        return value.to_list()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                pass
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
            return value
        return value
    return value


def parse_content_blocks(content: Any) -> list[Any]:
    """Parse content into a list of content blocks from mixed serialized formats."""
    content = normalize_value(content)
    if content is None:
        return []
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return [{"type": "text", "text": str(content)}]


def extract_text_from_content(content: Any) -> str:
    """Extract display text from structured content blocks."""
    blocks = parse_content_blocks(content)
    lines: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            lines.append(block)
            continue
        if not isinstance(block, dict):
            lines.append(str(block))
            continue

        block_type = block.get("type", "")
        if block_type == "text":
            lines.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            lines.append(
                f"[tool_use] {name} {json.dumps(tool_input, ensure_ascii=True)}"
            )
        elif block_type == "tool_result":
            lines.append(f"[tool_result] {block.get('content', '')}")
        else:
            if "text" in block:
                lines.append(str(block["text"]))
            elif "content" in block:
                lines.append(str(block["content"]))
            else:
                lines.append(json.dumps(block, ensure_ascii=True))
    return "\n".join(line for line in lines if line)


def normalize_tool_calls(tool_calls: Any) -> list[Any]:
    """Normalize tool call payloads and parse function.arguments JSON strings."""
    normalized = normalize_value(tool_calls)
    if not isinstance(normalized, list):
        return []

    out: list[Any] = []
    for call in normalized:
        call = normalize_value(call)
        if isinstance(call, dict):
            call_obj = dict(call)
            fn = call_obj.get("function")
            if isinstance(fn, dict):
                fn_obj = dict(fn)
                args = fn_obj.get("arguments")
                if isinstance(args, str):
                    try:
                        fn_obj["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        fn_obj["arguments"] = args
                call_obj["function"] = fn_obj
            out.append(call_obj)
        else:
            out.append(call)
    return out
