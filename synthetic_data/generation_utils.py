import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, cast

from anthropic.types.message_param import MessageParam
from datasets import Dataset, concatenate_datasets
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    Message,
)

MAX_RETRIES = 10

# Used for role playing tasks
GenerationRole = Literal["dungeon_master", "player", "adventure_parameters"]

RemoteModel = Literal[
    "claude-4-sonnet",
    "claude-4-5-sonnet",
    "claude-4-5-haiku",
    "claude-3-5-haiku",
    "qwen-qwq",
    "deepseek-v3",
    "deepseek-r1",
    "mistral-small-3",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "o3-mini",
    "mock",
    "vllm",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gpt-5-mini",
]


def convert_openai_to_anthropic_messages(
    conversation: list[dict[str, Any]],
) -> list[MessageParam]:
    """
    Convert OpenAI-style conversation format to Anthropic format.

    OpenAI uses:
    - {"role": "assistant", "content": "...", "tool_calls": [...]}
    - {"role": "tool", "tool_call_id": "...", "content": "..."}

    Anthropic uses:
    - {"role": "assistant", "content": [{"type": "tool_use", "id": "...", "name": "...", "input": {...}}]}
    - {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
    """
    converted: list[MessageParam] = []

    for msg in conversation:
        role = msg.get("role")

        if role == "tool":
            # Convert OpenAI tool result to Anthropic format
            converted.append(
                cast(
                    MessageParam,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id"),
                                "content": msg.get("content", ""),
                            }
                        ],
                    },
                )
            )
        elif role == "assistant" and "tool_calls" in msg:
            # Convert OpenAI assistant with tool_calls to Anthropic format
            content_blocks: list[dict[str, Any]] = []

            # Add text content if present
            if msg.get("content"):
                content_blocks.append(
                    {
                        "type": "text",
                        "text": msg["content"],
                    }
                )

            # Add tool_use blocks
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                tool_input = tc.get("function", {}).get("arguments", "{}")
                if isinstance(tool_input, str):
                    tool_input = json.loads(tool_input)
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": tc.get("function", {}).get("name"),
                        "input": tool_input,
                    }
                )

            converted.append(
                cast(
                    MessageParam,
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    },
                )
            )
        else:
            # Pass through other messages unchanged
            converted.append(cast(MessageParam, msg))

    return converted


def save_output_dataset(
    output_dataset: Dataset,
    dataset_name: str,
    new_dataset_rows: list[Dict],
    format: DatasetFormat,
    dataset_output_dir: str | None = None,
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    concatted_dataset = concatenate_datasets([output_dataset, dataset_new_rows])
    logger.info(
        f"Saving {len(new_dataset_rows)} new rows to {dataset_name}, {len(concatted_dataset)} total."
    )

    if format == DatasetFormat.HF_DATASET:
        concatted_dataset.push_to_hub(dataset_name)
    elif format == DatasetFormat.PARQUET:
        filename = f"{dataset_name}.parquet"
        if dataset_output_dir is None:
            raise ValueError(
                f"dataset_output_dir is required for {DatasetFormat.PARQUET} format"
            )
        dataset_path = os.path.join(dataset_output_dir, filename)
        logger.info(f"Saving to {dataset_path}")
        concatted_dataset.to_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported output format: {format}")


SHAREGPT_TO_OPENAI_ROLE = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


class GenerationArgs(BaseModel):
    """
    Arguments that can be overridden on a per-generation basis.
    """

    max_tokens: int = 4096
    temperature: float | None = None
    stop: list[str] | None = None
    seed: Optional[int] = None
    n_retries: int = MAX_RETRIES
    prefill: str | None = None
    thinking_budget: int | None = None
    tools: list[ChatCompletionToolParam] | None = None


FinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


@dataclass
class GenerationResult:
    """Result from a generation call that may include tool calls."""

    content: str | None
    tool_calls: list[ChatCompletionMessageToolCall] = field(default_factory=list)
    finish_reason: FinishReason = "end_turn"

    @property
    def message(self) -> Message:
        """Return a Message dict for compatibility with code expecting message format."""
        msg: Message = {
            "role": "assistant",
            "content": self.content or "",
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class GenWrapperArgs(BaseModel):
    """
    Properties of a specific model.
    """

    model_id: Optional[str] = None
    lora_name: Optional[str] = None
    default_generation_args: GenerationArgs = GenerationArgs()
    max_concurrent: int = 8
    max_rps: float = 8.0
    providers: list[str] | None = None
    is_reasoning_model: bool = False
    use_async_client: bool = True


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    provider_name: str

    def __init__(self, default_args: GenWrapperArgs) -> None:
        super().__init__()
        self.gen_wrapper_args = default_args

    @abstractmethod
    async def generate(
        self, conversations: list[Conversation], args: GenerationArgs | None = None
    ) -> list[GenerationResult]:
        """
        Generate completions for the given conversations.
        If args is provided, it will override the args in the wrapper.
        Tools can be provided via args.tools for tool-calling support.
        Returns GenerationResult objects that may contain content and/or tool calls.
        """
        pass

    def set_max_concurrent(self, max_concurrent: int) -> None:
        self.gen_wrapper_args.max_concurrent = max_concurrent
