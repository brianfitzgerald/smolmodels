import asyncio
import os
import traceback
from dataclasses import dataclass, fields
from typing import Optional, cast

from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic import AnthropicError, AsyncAnthropic
from anthropic import NotGiven as AnthropicNotGiven
from anthropic.types.message import Message as AnthropicMessage
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, LengthFinishReasonError, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from synthetic_data.generation_utils import (
    MAX_RETRIES,
    GenerationArgs,
    GenerationResult,
    GenerationWrapper,
    GenWrapperArgs,
    RemoteModel,
)
from synthetic_data.utils import (
    ContentBlock,
    Conversation,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    get_class_name,
)

MAX_TOOL_ITERATIONS = 10  # Prevent infinite tool loops

MOCK_SNIPPET = """
def solution(problem_input):
    return []
"""


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
        self, conversation: Conversation, args: GenerationArgs | None = None
    ) -> list[GenerationResult]:
        self.n_retries = MAX_RETRIES
        args = args or self.gen_wrapper_args.default_generation_args

        # Convert tools to OpenAI format if provided
        openai_tools = NOT_GIVEN
        if args.tools:
            openai_tools = args.tools  # type: ignore[assignment]
        while True:
            completion_requests = []
            temperature = args.temperature
            if self.gen_wrapper_args.is_reasoning_model:
                temperature = NOT_GIVEN

            request = self.oai_client.chat.completions.create(
                model=self.model_name,
                messages=conversation,  # type: ignore[arg-type]
                temperature=temperature,
                max_completion_tokens=args.max_tokens,
                extra_body=self.extra_body,
                seed=args.seed,
                tools=openai_tools,
            )
            completion_requests.append(request)

            try:
                results: list[ChatCompletion] = await asyncio.gather(
                    *completion_requests
                )
                if not results:
                    logger.error(results)
                    raise ValueError("No completions returned")

                generation_results = []
                for result in results:
                    if not result.choices:
                        continue

                    choice = result.choices[0]
                    oai_message = choice.message
                    content = oai_message.content or ""

                    # Prepend prefill to content for consistency with Anthropic behavior
                    if args.prefill and content:
                        content = args.prefill + content

                    text_block: TextBlock = {"type": "text", "text": content}
                    message: Message = {
                        "role": "assistant",
                        "content": [text_block],
                    }

                    if oai_message.tool_calls:
                        for tool_call in oai_message.tool_calls:
                            tool_use_block: ToolUseBlock = {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": tool_call.function.arguments,
                            }
                            message["content"].append(tool_use_block)

                    # Map OpenAI finish reasons to our internal format
                    finish_reason_map = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "content_filter": "refusal",
                        "function_call": "tool_use",
                    }
                    finish_reason = finish_reason_map.get(
                        choice.finish_reason or "stop", "end_turn"
                    )

                    generation_results.append(
                        GenerationResult(
                            message=message,
                            finish_reason=finish_reason,  # type: ignore[arg-type]
                        )
                    )

                return generation_results

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
        models_list = await self.oai_client.models.list()  # ty:ignore[invalid-await]
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
        self, conversation: Conversation, args: GenerationArgs | None = None
    ) -> list[GenerationResult]:
        args = args or self.gen_wrapper_args.default_generation_args
        try:
            assert self.gen_wrapper_args.model_id is not None, (
                "model_id is required for AnthropicGenerationWrapper"
            )

            # Extract system message if present
            system_msg = ANTHROPIC_NOT_GIVEN
            if conversation and conversation[0].get("role") == "system":
                first_block = conversation[0].get("content", [{}])[0]
                if first_block.get("type") == "text":
                    text_block: TextBlock = first_block  # type: ignore[assignment]
                    system_msg = text_block["text"]
                conversation = conversation[1:]

            # Setup working conversation with optional prefill
            conversation: list[Message] = list(conversation)
            if args.prefill:
                prefill_message: Message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": args.prefill}],  # ty:ignore[invalid-argument-type]
                }
                conversation.append(prefill_message)

            # Convert tools to Anthropic format if provided
            anthropic_tools: list | AnthropicNotGiven = ANTHROPIC_NOT_GIVEN
            if args.tools:
                anthropic_tools = [
                    {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "input_schema": t.get("input_schema", {}),
                    }
                    for t in args.tools
                ]

            # Tool execution loop
            all_content: list[ContentBlock] = []
            result: AnthropicMessage | None = await self.client.messages.create(
                model=self.gen_wrapper_args.model_id,
                messages=conversation,  # type: ignore[arg-type]
                system=system_msg,
                temperature=args.temperature
                if args.temperature is not None
                else ANTHROPIC_NOT_GIVEN,
                max_tokens=args.max_tokens,
                tools=anthropic_tools,
            )

            # Parse response content
            message_content: list[ContentBlock] = []
            for block in result.content:
                if block.type == "text":
                    text_block_content: TextBlock = {"type": "text", "text": block.text}
                    message_content.append(text_block_content)
                elif block.type == "tool_use":
                    tool_use: ToolUseBlock = {
                        "type": "tool_use",
                        "id": block.id,  # type: ignore[attr-defined]
                        "name": block.name,  # type: ignore[attr-defined]
                        "input": block.input,  # type: ignore[attr-defined]
                    }
                    message_content.append(tool_use)

            # Handle tool execution
            if result.stop_reason == "tool_use" and args.tool_use_executor:
                assistant_msg: Message = {
                    "role": "assistant",
                    "content": message_content,
                }

                # Execute tools and collect results
                tool_result_blocks: list[ContentBlock] = []
                for block in message_content:
                    if block.get("type") == "tool_use":
                        try:
                            tool_block = cast(ToolUseBlock, block)
                            result_content = args.tool_use_executor(tool_block)
                            tool_result: ToolResultBlock = {
                                "type": "tool_result",
                                "tool_use_id": tool_block["id"],
                                "content": result_content,
                            }
                            tool_result_blocks.append(tool_result)
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            error_result: ToolResultBlock = {
                                "type": "tool_result",
                                "tool_use_id": tool_block["id"],
                                "content": f"Error: {e}",
                            }
                            tool_result_blocks.append(error_result)

                user_msg: Message = {
                    "role": "user",
                    "content": tool_result_blocks,
                }

            return [
                GenerationResult(
                    message={"role": "assistant", "content": all_content or []},
                    finish_reason=result.stop_reason or "end_turn",
                )
            ]

        except AnthropicError as e:
            logger.error(f"Error while generating: {e}")
            return []


GEMINI_ROLE_MAP = {
    "system": "model",
    "user": "user",
    "assistant": "model",
}


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
    "claude-4-5-sonnet": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-sonnet-4-5-20250929"),
    ),
    "claude-3-5-haiku": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-3-5-haiku-latest"),
    ),
    "claude-4-5-haiku": RemoteModelChoice(
        AnthropicGenerationWrapper,
        GenWrapperArgs(max_rps=100, model_id="claude-haiku-4-5-20251001"),
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
    "vllm": RemoteModelChoice(
        VLLMWrapper,
        GenWrapperArgs(max_rps=8.0),
    ),
    "o3-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="o3-mini", max_rps=5000 / 60, is_reasoning_model=True),
    ),
    "gpt-5-mini": RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-5-mini", max_rps=5000 / 60),
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
