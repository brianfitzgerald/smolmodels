from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from synthetic_data.utils import (
    Conversation,
    Message,
    ToolParam,
    ToolResultBlock,
    ToolUseBlock,
)

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


class GenerationArgs(BaseModel):
    """
    Arguments that can be overridden on a per-generation basis.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    n_retries: int = 10
    tools: list[ToolParam] | None = None
    tool_use_executor: Callable[[ToolUseBlock], ToolResultBlock] | None = None


FinishReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "error",
]


@dataclass
class GenerationResult:
    added_messages: list[Message]
    finish_reason: FinishReason = "end_turn"


class GenWrapperArgs(BaseModel):
    """
    Properties of a specific model.
    """

    model_id: Optional[str] = None
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
        self, conversation: Conversation, args: GenerationArgs | None = None
    ) -> GenerationResult:
        """
        Generate completions for the given conversations.
        If args is provided, it will override the args in the wrapper.
        Tools can be provided via args.tools for tool-calling support.
        Returns GenerationResult objects that may contain content and/or tool calls.
        """
        pass

    def set_max_concurrent(self, max_concurrent: int) -> None:
        self.gen_wrapper_args.max_concurrent = max_concurrent
