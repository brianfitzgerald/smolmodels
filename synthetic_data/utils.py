import asyncio
from enum import Enum
import re
from typing import Dict, List, Optional, Union, Any
import json

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tabulate import tabulate
from pydantic.dataclasses import dataclass
from pydantic import Json


Conversation = List[ChatCompletionMessageParam]
ShareGPTConversation = List[Dict[str, str]]

JSONSchemaKey = Union[str, int, float, bool, List[Any], Dict[str, Any]]
JSONSchema = Dict[str, JSONSchemaKey]


@dataclass
class SquadExtractiveQARow:
    id: str
    context: str
    json_schema: JSONSchema
    fields: List[str]


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


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    ANTHROPIC = "anthropic"


class SeedDataFormat(Enum):
    TSV = "tsv"
    HF_DATASET = "hf_dataset"
    # Synthetic means the data is generated from a synthetic source, so no initial data is loaded
    SYNTHETIC = "synthetic"


def clean_message(message: JSONSchemaKey):
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
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    return message


def print_result_dicts(
    results: List[JSONSchema],
):
    if len(results) == 0:
        print("No results found, skipping print.")
        return
    columns = list(results[0].keys())
    new_dataset_row_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]

    col_widths = [40] * len(columns)
    for i, column in enumerate(columns):
        col = results[0][column]
        if isinstance(col, str) and col.isdigit():
            col_widths[i] = 10

    print("\n\n")
    print(
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
    except SyntaxError as e:
        return False
    return True


def clean_example(text):
    cleaned_paragraph = re.sub(
        r"1\. Scenario:.*?Example API Call:|```.*?```", "", text, flags=re.DOTALL
    )
    return cleaned_paragraph.strip()


JSON_MATCH_PATTERN = r"{.*}"


def recursive_json_parse(data: str) -> Optional[Union[Dict, str]]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        return {key: recursive_json_parse(value) for key, value in data.items()}
    return data


def extract_json(msg: str) -> Optional[JSONSchema]:
    """
    Parse out JSON from a string, and return the parsed JSON object.
    Works even if the JSON is embedded deep in a string or with recursive serialization.
    """
    msg = msg.replace("'", "").replace("\n", "")
    match = re.search(JSON_MATCH_PATTERN, msg)

    if match:
        json_str = match.group(0)
        json_obj = recursive_json_parse(json_str)
        return json_obj  # type: ignore
    return None


async def gather_with_concurrency_limit(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
