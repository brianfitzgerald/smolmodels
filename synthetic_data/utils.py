import ast
import asyncio
import json
import re
import shutil
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from loguru import logger
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic.dataclasses import dataclass
from tabulate import tabulate

Conversation = List[ChatCompletionMessageParam]
ShareGPTConversation = List[Dict[str, str]]

JSONSchemaKey = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
JSONSchema = Dict[str, JSONSchemaKey]

EvalDataModeChoice = Literal["random", "fixed"]


@dataclass
class ExtractiveQARow:
    context: str
    json_query: JSONSchema
    json_data: JSONSchema


@dataclass
class ToolFormerRow:
    question: str
    call_result: str
    tool_call: str
    agent_output: str


@dataclass
class ToolFormerDPORow:
    question: str
    call_result_accepted: str
    tool_call_accepted: str
    agent_output_accepted: str
    call_result_rejected: str
    tool_call_rejected: str
    agent_output_rejected: str


@dataclass
class SyntheticToolCallRow:
    tool: str
    question: str
    tool_call: str
    call_result: str
    agent_output: str


@dataclass
class SyntheticToolCallDPORow:
    tool: str
    question: str
    tool_call_accepted: str
    call_result_accepted: str
    agent_output_accepted: str
    tool_call_rejected: str
    call_result_rejected: str
    agent_output_rejected: str


class DatasetFormat(Enum):
    TSV = "tsv"
    HF_DATASET = "hf_dataset"
    # Synthetic means the data is generated from a synthetic source, so no initial data is loaded
    SYNTHETIC = "synthetic"
    PARQUET = "parquet"
    CUSTOM = "custom"
    NONE = "none"


def clean_message(message: JSONSchemaKey, truncate_length: int | None = None):
    """
    Clean up spaces, tabs, and newlines in a message with a JSON dict, so the dict is formatted nicely.
    """

    if isinstance(message, list):
        message = ", ".join(message)
    elif isinstance(message, bool):
        message = str(message)
    elif isinstance(message, (int, float)):
        message = str(message)
    elif isinstance(message, dict):
        message = json.dumps(message, indent=2)

    # Handle odd edge case where textwrap evaluates the value as a bool
    if message == "True" or message == "False":
        message = message.lower()
    if message is None:
        message = ""
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    if truncate_length is not None and len(message) > truncate_length:
        message = message[:truncate_length] + "..."
    return message


def print_result_dicts(
    results: List[JSONSchema],
):
    if len(results) == 0:
        logger.warning("No results found, skipping print.")
        return
    columns = list(results[0].keys())
    new_dataset_row_elements = [
        [clean_message(row[column], truncate_length=1000) for column in columns]
        for row in results
    ]

    col_widths = [40] * len(columns)
    for i, column in enumerate(columns):
        col = results[0][column]
        if isinstance(col, str) and col.isdigit():
            col_widths[i] = 10

    logger.info(
        tabulate(
            new_dataset_row_elements,
            headers=columns,
            tablefmt="simple_grid",
            maxcolwidths=col_widths,
        )
    )


def extract_code_blocks(text):
    pattern = r"```(?:.*?)```|<code>(?:.*?)</code>"

    code_blocks = re.findall(pattern, text, re.DOTALL)

    clean_code_blocks = [
        block.strip("`").strip("<code>").strip("</code>").strip()
        for block in code_blocks
    ]

    code_blocks_str = "\n".join(clean_code_blocks)
    return code_blocks_str


LINE_REFIX_PATTERN = re.compile(
    r"(User:|Task:|API:|Call:|Result:|Agent:)\s*(.*)", re.IGNORECASE
)


def get_matches(text: str):
    matches = LINE_REFIX_PATTERN.findall(text)

    extracted_lines = []
    for match in matches:
        if match[1]:
            extracted_lines.append(match[1])
    return extracted_lines


# TODO refactor this into a base class with methods for prompting, parsing, etc.
# Would also have properties used for the dataclass
# don't want to do this yet until we have the full flow working


def extract_toolformer_row(text: str) -> ToolFormerRow:
    question, tool_call, call_result, agent_output = get_matches(text)
    return ToolFormerRow(question, call_result, tool_call, agent_output)


def is_valid_python(json_str: str):
    json_str = json_str.strip().replace("`", "")
    try:
        compile(json_str, "<string>", "single")
    except SyntaxError:
        traceback.print_exc()
        return False
    return True


def clean_example(text):
    cleaned_paragraph = re.sub(
        r"1\. Scenario:.*?Example API Call:|```.*?```", "", text, flags=re.DOTALL
    )
    return cleaned_paragraph.strip()


def recursive_json_parse(data: str) -> Optional[Union[Dict, str]]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        return {key: recursive_json_parse(value) for key, value in data.items()}
    return data


JSON_MATCH_PATTERN = r"```(?:json)?\n(.*?)\n```"


def extract_json_code_blocks(msg: str) -> List[JSONSchema]:
    """
    Parse out JSON code blocks from Markdown or plain text.
    Works even if the JSON is embedded deep in a string or with recursive serialization.
    """
    blocks = re.findall(JSON_MATCH_PATTERN, msg, re.DOTALL)

    res = []
    for match in blocks:
        match = match.strip()
        if match:
            json_obj = recursive_json_parse(match)
            res.append(json_obj)
    return res


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)


def extract_code_block(msg: str, language: str = "python") -> List[str]:
    """
    Extract a code block from a message. If none are found, treat the entire message as a code block.
    """
    match_pattern = rf"```(?:{language})?\n(.*?)\n```"
    blocks = re.findall(match_pattern, msg, re.DOTALL)

    if not blocks:
        msg = msg.lstrip("assistant\n\n")
        try:
            ast.parse(msg)
        except SyntaxError:
            return []
        blocks = [msg]

    return blocks


def extract_text_between_tags(input_text: str, tag_name: str):
    pattern = rf"<{tag_name}>(.*?)(?:</{tag_name}>|$)"
    return re.findall(pattern, input_text, re.DOTALL)


async def gather_with_concurrency_limit(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def get_class_name(obj):
    return obj.__class__.__name__


def ldictl(dict_of_lists: List[dict]):
    """
    List of dicts to dict of lists.
    """
    return (
        {key: [d[key] for d in dict_of_lists] for key in dict_of_lists[0]}
        if dict_of_lists
        else {}
    )


def dictl(dict_of_lists: dict) -> list[dict]:
    """
    Dict of lists to list of dicts.
    """
    return [dict(zip(dict_of_lists.keys(), t)) for t in zip(*dict_of_lists.values())]


def chunk_list(xs: List, n: int):
    n = max(1, n)
    return (xs[i : i + n] for i in range(0, len(xs), n))


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten sublists
        else:
            flat_list.append(item)
    return flat_list


def log_to_file(
    all_eval_rows: list, new_rows_to_log: list, output_dir: str, current_step: int
):
    ensure_directory(output_dir, clear=False)
    all_rows_pd = pd.DataFrame(all_eval_rows)
    all_rows_pd.to_parquet(f"{output_dir}/eval_samples.parquet")

    tabulate_str = tabulate(
        new_rows_to_log,
        headers="keys",
        tablefmt="simple_grid",
        maxcolwidths=[50] * 5,
    )
    print(tabulate_str)
    with open(f"{output_dir}/eval_samples.txt", "a") as f:
        f.write("\n" * 2)
        f.write(
            "".join(
                ["#" for _ in range(40)]
                + [f"  Eval samples for step: {current_step}  "]
                + ["#" for _ in range(40)]
                + ["\n" * 2]
            )
        )
        f.write(tabulate_str)

    logger.info("Saved eval samples.")


def log_conversation(conversation: Conversation) -> None:
    """
    Log a Conversation to the console using loguru with colored output based on roles.

    Args:
        conversation: The conversation to log
    """
    # Color mapping for different roles using ANSI color codes
    role_colors = {
        "system": "\033[36m",  # Cyan
        "user": "\033[32m",  # Green
        "assistant": "\033[34m",  # Blue
        "function": "\033[33m",  # Yellow
        "tool": "\033[35m",  # Magenta
        "unknown": "\033[31m",  # Red
    }
    reset_color = "\033[0m"
    white_color = "\033[37m"

    # Build the complete log message
    log_lines = []
    log_lines.append(f"\n{'=' * 60}")
    log_lines.append(f"{white_color}CONVERSATION LOG{reset_color}")
    log_lines.append(f"{'=' * 60}")

    for i, message in enumerate(conversation, 1):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if isinstance(content, str):
            content_str = content
        elif content is None:
            content_str = ""
        else:
            content_str = str(content)

        # Get color for role, default to red for unknown roles
        color = role_colors.get(role, role_colors["unknown"])

        # Add the message with color and clear separation
        log_lines.append(f"\n{color}[{i}] {role.upper()}{reset_color}\n")
        log_lines.append(f"{color}{content_str}{reset_color}\n")

    logger.info("".join(log_lines))
    print("".join(log_lines))


def parse_xml_tags(text: str, required_tags: List[str] = []) -> Dict[str, str]:
    result = {}
    missing_tags = []

    # First, find all XML tags in the text
    import re

    tag_pattern = r"<([^>]+)>"
    all_tags: set[str] = set(re.findall(tag_pattern, text))

    # Filter out closing tags and self-closing tags
    opening_tags = set()
    for tag in all_tags:
        if not tag.startswith("/") and not tag.endswith("/"):
            opening_tags.add(tag)

    # Parse all found tags
    for tag in opening_tags:
        opening_tag = f"<{tag}>"
        closing_tag = f"</{tag}>"

        # Find the tags (case-insensitive search)
        text_lower = text.lower()
        opening_tag_lower = opening_tag.lower()
        closing_tag_lower = closing_tag.lower()

        start_idx = text_lower.find(opening_tag_lower)
        end_idx = text_lower.find(closing_tag_lower)

        if start_idx != -1 and end_idx != -1:
            # Extract content using original case text
            content_start = start_idx + len(opening_tag)
            content_end = end_idx
            content = text[content_start:content_end].strip()
            result[tag] = content

    # Check for missing required tags
    for tag in required_tags:
        if tag not in result:
            missing_tags.append(tag)

    if missing_tags:
        raise ValueError(f"Required XML tags not found: {', '.join(missing_tags)}")

    return result
