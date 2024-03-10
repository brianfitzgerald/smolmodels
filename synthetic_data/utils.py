import ast
import asyncio
import re
from typing import Dict, List
from pydantic import BaseModel

from tabulate import tabulate
from pydantic.dataclasses import dataclass


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
class ToolUsageDPORow:
    definition: str
    task: str
    tool_call: str
    call_result: str
    agent_output: str


def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message, so the JSON is formatted nicely.
    """

    # Handle odd edge case where textwrap evaluates the value as a bool
    if message == "True" or message == "False":
        message = message.lower()
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    return message


def print_result_dicts(
    results: List[Dict],
):
    if len(results) == 0:
        print("No results found, skipping print.")
        return
    columns = list(results[0].keys())
    new_dataset_row_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]

    col_widths = [35] * len(columns)
    for i, column in enumerate(columns):
        if results[0][column].isdigit():
            col_widths[i] = 10

    print(
        tabulate(
            new_dataset_row_elements,
            headers=columns,
            tablefmt="simple",
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


def extract_toolformer_dpo_row(
    text: str, original_row: ToolFormerRow
) -> ToolFormerDPORow:
    tool_call, call_result, agent_output = get_matches(text)
    return ToolFormerDPORow(
        original_row.question,
        original_row.call_result,
        original_row.tool_call,
        original_row.agent_output,
        call_result,
        tool_call,
        agent_output,
    )


def extract_tool_usage_dpo_row(text: str) -> ToolUsageDPORow:
    definition, task, tool_call, call_result, agent_output = get_matches(text)
    return ToolUsageDPORow(definition, task, tool_call, call_result, agent_output)


def assert_valid_python_code(json_str: str):
    json_str = json_str.strip().replace("`", "")
    try:
        compile(json_str, "<string>", "single")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code: {e}")


def clean_example(text):
    cleaned_paragraph = re.sub(
        r"1\. Scenario:.*?Example API Call:|```.*?```", "", text, flags=re.DOTALL
    )
    return cleaned_paragraph.strip()


async def gather_with_concurrency_limit(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
