import asyncio
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional, cast

import google.genai as genai
import google.genai.types
from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types.message import Message
from anthropic.types.message_param import MessageParam
from datasets import Dataset, concatenate_datasets
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, LengthFinishReasonError, NotGiven, OpenAI
from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN, NotGiven as AnthropicNotGiven
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    get_class_name,
)

SHAREGPT_TO_OPENAI_ROLE = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


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


MAX_RETRIES = 10


class GenerationArgs(BaseModel):
    """
    Arguments that can be overridden on a per-generation basis.
    """

    max_tokens: int = 4096
    temperature: float = 0.4
    stop: list[str] | None = None
    seed: Optional[int] = None
    n_retries: int = MAX_RETRIES
    prefill: str | None = None
    thinking_budget: int | None = None


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
    ) -> list[str]:
        """
        Generate completions for the given conversations.
        If args is provided, it will override the args in the wrapper.
        """
        pass

    def set_max_concurrent(self, max_concurrent: int) -> None:
        self.gen_wrapper_args.max_concurrent = max_concurrent


MOCK_SNIPPET = """
def solution(problem_input):
    return []
"""


class MockGenerator(GenerationWrapper):
    def __init__(self, _: GenWrapperArgs) -> None:
        self.mock_completions = []

    def set_mock_completions(self, completions: list[str]) -> None:
        self.mock_completions = completions

    async def generate(
        self, conversations: list[Conversation], args: GenerationArgs | None = None
    ) -> list[str]:
        if self.mock_completions:
            return self.mock_completions
        return [MOCK_SNIPPET] * len(conversations)


class OpenAIGenerationWrapper(GenerationWrapper):
    provider_name: str = "openai"

    def __init__(
        self,
        default_args: GenWrapperArgs,
        key_env_var_name: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(default_args)
        api_key = os.environ.get(key_env_var_name)
        if api_key is None:
            raise ValueError(
                f"{key_env_var_name} is required for {get_class_name(self)}"
            )
        if default_args.use_async_client:
            self.oai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.oai_client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = default_args.model_id or "gpt-4o-mini"
        self.extra_body = {}

    async def generate(
        self, conversations: list[Conversation], args: GenerationArgs | None = None
    ) -> list[str]:
        self.n_retries = MAX_RETRIES
        args = args or self.gen_wrapper_args.default_generation_args
        while True:
            completion_requests = []
            for conversation in conversations:
                temperature = args.temperature
                if self.gen_wrapper_args.is_reasoning_model:
                    temperature = NOT_GIVEN

                # TODO re add structured output support
                request = self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    temperature=temperature,
                    max_completion_tokens=args.max_tokens,
                    extra_body=self.extra_body,
                    seed=args.seed,
                )
                completion_requests.append(request)
            try:
                results: list[ChatCompletion] = await asyncio.gather(
                    *completion_requests
                )
                if not results:
                    logger.error(results)
                    raise ValueError("No completions returned")
                assert all(len(result.choices) > 0 for result in results), (
                    "No completions returned"
                )
                completions = [
                    result.choices[0].message.content
                    for result in results
                    if result.choices[0].message.content is not None
                ]
                return completions
            except LengthFinishReasonError as e:
                logger.error(f"Max length error: {e}")
                return []
            except Exception as e:
                logger.error(
                    f"Error while generating: {e}, retries left: {self.n_retries}"
                )
                traceback.print_exc()
                await asyncio.sleep(1)
                self.n_retries -= 1
                if self.n_retries <= 0:
                    raise e


def _get_model_id(base_url: str):
    logger.info("Fetching models from OpenAI client")
    sync_client = OpenAI(base_url=base_url)
    all_models = sync_client.models.list()

    logger.info(f"Models: {[x.id for x in all_models.data]}")
    return all_models.data[0].id


class VLLMWrapper(OpenAIGenerationWrapper):
    provider_name: str = "vllm"

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args)
        base_url = "http://localhost:8000/v1"
        self.oai_client = AsyncOpenAI(
            base_url=base_url,
        )
        self.temperature = 0.4
        self.model_name = args.model_id
        logger.info(f"Using model: {self.model_name}")
        # If true, list models and call the first one listed
        self.n_retries = MAX_RETRIES
        self.temperature = 0.2
        self.max_tokens = 4096
        self.args = args
        if args.model_id is None:
            logger.info(f"Using model: {self.model_name}")
            self.model_name = _get_model_id(base_url)

    async def _list_models(self):
        models_list = await self.oai_client.models.list()
        model_ids = [m.id for m in models_list.data]
        logger.info(f"Models: {model_ids}")
        self.model_name = model_ids[0]
        logger.info(f"Using model ID: {self.model_name}")


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
        if args.providers:
            logger.info(f"Using providers: {args.providers}")
            self.extra_body["provider"] = {"order": args.providers}


class AnthropicGenerationWrapper(GenerationWrapper):
    provider_name = "anthropic"

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for AnthropicGenerationWrapper"
            )
        self.client = AsyncAnthropic(
            api_key=api_key,
        )

    async def generate(
        self, conversations: list[Conversation], args: GenerationArgs | None = None
    ) -> list[str]:
        args = args or self.gen_wrapper_args.default_generation_args
        try:
            completion_requests = []
            for conversation in conversations:
                assert self.gen_wrapper_args.model_id is not None, (
                    "model_id is required for AnthropicGenerationWrapper"
                )
                system_msg: str | AnthropicNotGiven = ANTHROPIC_NOT_GIVEN
                conversation = cast(list[MessageParam], conversation)
                if conversation[0]["role"] == "system":
                    system_msg = conversation[0]["content"]  # type: ignore
                    conversation = conversation[1:]
                if args.prefill:
                    conversation = [
                        *conversation,
                        {"role": "assistant", "content": args.prefill},
                    ]
                request = self.client.messages.create(
                    model=self.gen_wrapper_args.model_id,
                    messages=conversation,
                    system=system_msg,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                completion_requests.append(request)
            results: list[Message] = await asyncio.gather(*completion_requests)
            completions = [
                result.content[0].text  # type: ignore
                for result in results
                if result.content is not None
            ]
            if args.prefill:
                completions = [args.prefill + completion for completion in completions]
            return completions
        except AnthropicError as e:
            logger.error(f"Error while generating: {e}")
            return []


GEMINI_ROLE_MAP = {
    "system": "model",
    "user": "user",
    "assistant": "model",
}


def _openai_conversation_to_gemini(conversation: Conversation):
    return [
        google.genai.types.Content(
            role=GEMINI_ROLE_MAP[message["role"]],
            parts=[google.genai.types.Part.from_text(text=message["content"])],  # type: ignore
        )
        for message in conversation
    ]


class GeminiWrapper(GenerationWrapper):
    provider_name = "gemini"

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args)
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.model_id = args.model_id or "gemini-2.0-flash"
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY is required for GeminiWrapper")
        self.client = genai.Client(api_key=api_key)
        self.args = args

    async def generate(
        self, conversations: list[Conversation], args: GenerationArgs | None = None
    ) -> list[str]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create new requests for each attempt
                reqs = []
                for conv in [_openai_conversation_to_gemini(c) for c in conversations]:
                    reqs.append(
                        self.client.aio.models.generate_content(
                            model=self.model_id,
                            contents=[conv]
                        )
                    )

                results: list[
                    google.genai.types.GenerateContentResponse
                ] = await asyncio.gather(*reqs)

                # Check for null texts
                null_texts = [r for r in results if r.text is None]
                if null_texts:
                    logger.error(
                        f"Null texts found in attempt {attempt + 1}: {null_texts}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    return []

                return [result.text or "" for result in results]

            except Exception as e:
                logger.error(f"Error while generating (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return []

        # Return empty list if all retries failed
        return []


GenerationRole = Literal["generation", "followup", "parameter"]

RemoteModel = Literal[
    "claude-4-sonnet",
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
]


@dataclass
class RemoteModelChoice:
    model: type[GenerationWrapper]
    args: Optional[GenWrapperArgs] = None


MODEL_CONFIGS: dict[RemoteModel, RemoteModelChoice] = {
    "qwen-qwq": RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(model_id="qwen/qwq-32b-preview", max_rps=500),
    ),
    "deepseek-v3": RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(model_id="deepseek/deepseek-chat", max_rps=500),
    ),
    "deepseek-r1": RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(
            model_id="deepseek/deepseek-r1", max_rps=500, providers=["Fireworks"]
        ),
    ),
    "mistral-small-3": RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(
            model_id="mistralai/mistral-small-24b-instruct-2501",
            max_rps=64,
            max_concurrent=8,
        ),
    ),
    "claude-4-sonnet": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-sonnet-4-20250514"),
    ),
    "claude-3-5-haiku": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-3-5-haiku-latest"),
    ),
    "gpt-4o-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4o-mini", max_rps=int(5000 / 60)),
    ),
    "gpt-4o": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4o", max_rps=5000 / 60),
    ),
    "gpt-4.1-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4.1-mini", max_rps=5000 / 60),
    ),
    "gpt-4.1-nano": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4.1-nano", max_rps=5000 / 60),
    ),
    "mock": RemoteModelChoice(MockGenerator),
    "vllm": RemoteModelChoice(
        VLLMWrapper,
        GenWrapperArgs(max_rps=8.0),
    ),
    "o3-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="o3-mini", max_rps=5000 / 60, is_reasoning_model=True),
    ),
    "gemini-2.0-flash": RemoteModelChoice(
        GeminiWrapper,
        GenWrapperArgs(model_id="gemini-2.0-flash", max_rps=5000 / 60),
    ),
    "gemini-2.5-flash": RemoteModelChoice(
        GeminiWrapper,
        GenWrapperArgs(model_id="gemini-2.5-flash", max_rps=5000 / 60),
    ),
}


def get_generation_wrapper(
    model_name: RemoteModel, args_override: GenWrapperArgs | None = None
) -> GenerationWrapper:
    config = MODEL_CONFIGS[model_name]
    if args_override:
        for field in fields(args_override):
            if (
                field.default != getattr(args_override, field.name)
                and field.name != "dotenv"
            ):
                setattr(config.args, field.name, getattr(args_override, field.name))
                logger.info(
                    f"Overriding {field.name} in gen wrapper args with {getattr(args_override, field.name)}"
                )
    elif config.args is None:
        config.args = GenWrapperArgs()
    assert config.args is not None
    return config.model(config.args)
