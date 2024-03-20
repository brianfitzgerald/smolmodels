import asyncio
from enum import Enum
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional, Literal


from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tabulate import tabulate
from pydantic.dataclasses import dataclass


Conversation = List[ChatCompletionMessageParam]
ShareGPTConversation = List[Dict[str, str]]


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


class DatasetTaskFormat(str, Enum):
    """
    Whether a task performs the DPO or SFT objectives.
    """

    SFT = "SFT"
    DPO = "DPO"


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    GROQ = "groq"


class DatasetFormat(Enum):
    CSV = "tsv"
    HF_DATASET = "hf_dataset"
    # Synthetic means the data is generated from a synthetic source, so no initial data is loaded
    SYNTHETIC = "synthetic"
    PARQUET = "parquet"


def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message with a JSON dict, so the dict is formatted nicely.
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
    printed_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]

    col_widths = [35] * len(columns)
    for i, column in enumerate(columns):
        if results[0][column].isdigit():
            col_widths[i] = 10

    print(
        tabulate(
            printed_elements,
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


async def gather_with_concurrency_limit(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    Duplicated between submodules to avoid importing torch
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)


LabelsSet = Literal[
    "clipdrop_synthetic", "i2p", "clipdrop_binary", "multilabel_safe_famous_figures"
]

SAFERPROMPT_LABELS = {
    "safe": 0,
    "famous_figures": 1,
    "nudity": 2,
    "sexual_content": 3,
    "violence": 4,
    "discriminatory_content": 5,
}

ANNOTATED_LABELS = {"safe": 0, "unsafe": 1, "borderline": 2}

FAMOUS_FIGURES_LABELS = {"negative": 0, "positive": 1}

I2P_LABELS = {
    "hate": 0,
    "harassment": 1,
    "violence": 2,
    "self-harm": 3,
    "sexual-content": 4,
    "shocking-images": 5,
    "illegal-activity": 6,
}

MULTILABEL_SAFE_FAMOUS_FIGURES = {"negative": 0, "positive": 1, "safe": 2, "unsafe": 3}

LABEL_SETS: Dict[LabelsSet, Dict] = {
    "clipdrop_synthetic": SAFERPROMPT_LABELS,
    "clipdrop_binary": ANNOTATED_LABELS,
    "i2p": I2P_LABELS,
    "multilabel_safe_famous_figures": MULTILABEL_SAFE_FAMOUS_FIGURES,
}

LABEL_REGEX = re.compile(r"Label:\s*(.*)")


def extract_label(text: str, valid_labels: Dict) -> Optional[str]:
    match = LABEL_REGEX.search(text)
    if match:
        label = match.group(1).strip().lower()
        # remove labels with multiple words
        if " " in label:
            return None
        if label not in valid_labels:
            return None
        return label
    else:
        return None
