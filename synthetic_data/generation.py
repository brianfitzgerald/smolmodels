import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import traceback
from typing import Dict, List, Optional, cast, Literal
from datetime import datetime, timedelta
import asyncio
from collections import deque

import google.genai as genai
import google.genai.types
from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types.message import Message
from anthropic.types.message_param import MessageParam
from datasets import Dataset, concatenate_datasets
from loguru import logger
from openai import NOT_GIVEN, LengthFinishReasonError, NotGiven, AsyncOpenAI, OpenAI
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
    new_dataset_rows: List[Dict],
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


@dataclass
class GenWrapperArgs:
    model_id: Optional[str] = None
    lora_name: Optional[str] = None
    max_concurrent: int = 8
    max_rps: float = 8.0
    max_tokens: int = 4096
    temperature: float = 0.4
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN
    n_retries: int = MAX_RETRIES
    providers: List[str] | None = None
    stop: List[str] | None = None
    is_reasoning_model: bool = False
    seed: Optional[int] = None
    async_client: bool = False


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__()
        self.args = args

    @abstractmethod
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        pass


MOCK_SNIPPET = """
def solution(problem_input):
    return []
"""


class MockGenerator(GenerationWrapper):
    def __init__(self, _: GenWrapperArgs) -> None:
        self.mock_completions = []

    def set_mock_completions(self, completions: List[str]) -> None:
        self.mock_completions = completions

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        if self.mock_completions:
            return self.mock_completions
        return [MOCK_SNIPPET] * len(conversations)


class RPSLimiter:
    def __init__(self, max_rps: float):
        self.max_rps = max_rps
        self.request_times = deque()

    async def acquire(self):
        now = datetime.now()

        # Remove old requests outside the 1 second window
        while self.request_times and (now - self.request_times[0]) > timedelta(
            seconds=1
        ):
            self.request_times.popleft()

        # If at RPS limit, wait until we can make another request
        if len(self.request_times) >= self.max_rps:
            wait_time = 1.0 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Waiting {wait_time} seconds to not exceed RPS limit")
                await asyncio.sleep(wait_time)

        self.request_times.append(now)


class OpenAIGenerationWrapper(GenerationWrapper):
    def __init__(
        self,
        args: GenWrapperArgs,
        key_env_var_name: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(args)
        api_key = os.environ.get(key_env_var_name)
        if api_key is None:
            raise ValueError(
                f"{key_env_var_name} is required for {get_class_name(self)}"
            )
        if args.async_client:
            self.oai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.oai_client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = args.model_id or "gpt-4o-mini"
        self.args = args
        self.rps_limiter = RPSLimiter(args.max_rps)
        self.extra_body = {}

    def generate_sync(self, conversations: List[Conversation]) -> List[str]:
        response: ChatCompletion = self.oai_client.chat.completions.create(
            model=self.model_name,
            messages=conversations,
            temperature=self.args.temperature,
            max_completion_tokens=self.args.max_tokens,
        )
        return [r.message.content for r in response.choices]

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        self.n_retries = MAX_RETRIES
        while True:
            completion_requests = []
            for conversation in conversations:
                await self.rps_limiter.acquire()

                temperature = self.args.temperature
                if self.args.is_reasoning_model:
                    temperature = NOT_GIVEN

                if self.args.response_format:
                    # have to use the beta provider
                    request = self.oai_client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_completion_tokens=self.args.max_tokens,
                        response_format=self.args.response_format,  # type: ignore
                        extra_body=self.extra_body,
                        seed=self.args.seed,
                    )
                else:
                    request = self.oai_client.chat.completions.create(
                        model=self.model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_completion_tokens=self.args.max_tokens,
                        extra_body=self.extra_body,
                        seed=self.args.seed,
                    )
                completion_requests.append(request)
            try:
                results: List[ChatCompletion] = await asyncio.gather(
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
    def __init__(self, args: GenWrapperArgs) -> None:
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
    def __init__(self, args: GenWrapperArgs) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for AnthropicGenerationWrapper"
            )
        self.client = AsyncAnthropic(
            api_key=api_key,
        )

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        try:
            completion_requests = []
            for conversation in conversations:
                conversation = cast(List[MessageParam], conversation)
                assert self.args.model_id is not None, (
                    "model_id is required for AnthropicGenerationWrapper"
                )
                request = self.client.messages.create(
                    model=self.args.model_id,
                    messages=conversation,
                    temperature=0,
                    max_tokens=4096,
                )
                completion_requests.append(request)
            results: List[Message] = await asyncio.gather(*completion_requests)
            completions = [
                result.content[0].text  # type: ignore
                for result in results
                if result.content is not None
            ]
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
            parts=[google.genai.types.Part(text=message["content"])],  # type: ignore
        )
        for message in conversation
    ]


class GeminiWrapper(GenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.model_id = args.model_id or "gemini-2.0-flash"
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY is required for GeminiWrapper")
        self.client = genai.Client(api_key=api_key)
        self.args = args

    async def generate(self, conversations: List[Conversation]):
        reqs = []
        for conv in [_openai_conversation_to_gemini(c) for c in conversations]:
            reqs.append(
                self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=conv,  # type: ignore
                )
            )
        try:
            results: List[
                google.genai.types.GenerateContentResponse
            ] = await asyncio.gather(*reqs)
        except Exception as e:
            logger.error(f"Error while generating: {e}")
            return []
        for r in results:
            if r.text is None:
                logger.error(f"Null text for: {r}")
        return [result.text for result in results]


RemoteModel = Literal[
    "claude-3-7",
    "claude-3-5",
    "qwen-qwq",
    "deepseek-v3",
    "deepseek-r1",
    "mistral-small-3",
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
    "mock",
    "vllm",
    "gemini-2.0-flash",
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
    "claude-3-5": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-3-5-sonnet-20241022"),
    ),
    "claude-3-7": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-3-7-sonnet-20250219"),
    ),
    "gpt-4o-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4o-mini", max_rps=int(5000 / 60)),
    ),
    "gpt-4o": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4o", max_rps=5000 / 60),
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
