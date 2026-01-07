import json
import os
import traceback
from dataclasses import dataclass, fields
from typing import Optional

from anthropic import AsyncAnthropic
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, LengthFinishReasonError, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from synthetic_data.generation_utils import (
    MAX_RETRIES,
    FinishReason,
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
    log_conversation,
)


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

    def _convert_tools_to_openai_format(self, tools: list) -> list:
        """Convert Anthropic-format tools to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _convert_conversation_to_openai_format(
        self, conversation: Conversation
    ) -> list:
        """Convert internal conversation format to OpenAI format.

        Internal format uses:
        - assistant messages with tool_use blocks
        - user messages with tool_result blocks

        OpenAI format uses:
        - assistant messages with tool_calls field (no tool_use in content)
        - tool role messages for tool results
        """
        openai_messages = []

        for msg in conversation:
            role = msg["role"]
            content = msg.get("content", [])

            if role == "assistant":
                # Check if there are tool_use blocks
                tool_uses = [
                    block for block in content if block.get("type") == "tool_use"
                ]
                text_blocks = [
                    block for block in content if block.get("type") == "text"
                ]

                if tool_uses:
                    # Create OpenAI tool_calls format
                    tool_calls = []
                    for tool_use in tool_uses:
                        tool_input = tool_use.get("input", {})
                        if not isinstance(tool_input, str):
                            tool_input = json.dumps(tool_input)

                        tool_calls.append(
                            {
                                "id": tool_use.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tool_use.get("name"),
                                    "arguments": tool_input,
                                },
                            }
                        )

                    # Create assistant message with tool_calls
                    oai_msg = {
                        "role": "assistant",
                        "content": text_blocks[0].get("text") if text_blocks else None,
                        "tool_calls": tool_calls,
                    }
                    openai_messages.append(oai_msg)
                else:
                    # Regular assistant message
                    text = text_blocks[0].get("text") if text_blocks else ""
                    openai_messages.append(
                        {
                            "role": "assistant",
                            "content": text,
                        }
                    )

            elif role == "user":
                # Check if there are tool_result blocks
                tool_results = [
                    block for block in content if block.get("type") == "tool_result"
                ]
                text_blocks = [
                    block for block in content if block.get("type") == "text"
                ]

                if tool_results:
                    # Create tool role messages for each result
                    for tool_result in tool_results:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result.get("tool_use_id"),
                                "content": tool_result.get("content", ""),
                            }
                        )
                else:
                    # Regular user message
                    text = text_blocks[0].get("text") if text_blocks else ""
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": text,
                        }
                    )

            else:  # system or other roles
                # Pass through
                if content:
                    text = content[0].get("text") if content else ""
                    openai_messages.append(
                        {
                            "role": role,
                            "content": text,
                        }
                    )

        return openai_messages

    async def generate(
        self, conversation: Conversation, args: GenerationArgs | None = None
    ) -> GenerationResult:
        args = args or self.gen_wrapper_args.default_generation_args
        max_tool_iterations = 10  # Prevent infinite loops
        iteration = 0

        try:
            # Setup working conversation with optional prefill
            if args.prefill:
                prefill_text_block: TextBlock = {"type": "text", "text": args.prefill}
                prefill_message: Message = {
                    "role": "assistant",
                    "content": [prefill_text_block],
                }
                conversation.append(prefill_message)

            # Convert tools to OpenAI format if provided
            openai_tools = NOT_GIVEN
            if args.tools:
                openai_tools = self._convert_tools_to_openai_format(args.tools)  # type: ignore[assignment]

            # Tool use loop: continue until we get a non-tool_use response
            while iteration < max_tool_iterations:
                iteration += 1

                temperature = args.temperature
                if self.gen_wrapper_args.is_reasoning_model:
                    temperature = NOT_GIVEN

                # Convert conversation to OpenAI format
                openai_conversation = self._convert_conversation_to_openai_format(
                    conversation
                )

                result: ChatCompletion = await self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_conversation,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_completion_tokens=args.max_tokens,
                    extra_body=self.extra_body,
                    seed=args.seed,
                    tools=openai_tools,
                )

                if not result.choices:
                    logger.error("No choices returned from OpenAI API")
                    return GenerationResult(
                        conversation=conversation, finish_reason="error"
                    )

                choice = result.choices[0]
                oai_message = choice.message

                # Build content blocks
                assistant_message_blocks: list[ContentBlock] = []

                # Add text content if present
                if oai_message.content:
                    content_text = oai_message.content
                    # Prepend prefill to content for consistency with Anthropic behavior
                    if args.prefill and iteration == 1:
                        content_text = args.prefill + content_text

                    text_block: TextBlock = {"type": "text", "text": content_text}
                    assistant_message_blocks.append(text_block)

                # Add tool calls if present
                if oai_message.tool_calls:
                    for tool_call in oai_message.tool_calls:
                        # Parse arguments if they're a string
                        tool_input = tool_call.function.arguments
                        if isinstance(tool_input, str):
                            import json

                            try:
                                tool_input = json.loads(tool_input)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse tool arguments: {tool_input}"
                                )

                        tool_use_block: ToolUseBlock = {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": tool_input,
                        }
                        assistant_message_blocks.append(tool_use_block)

                # Add the assistant message to the conversation
                if assistant_message_blocks:
                    assistant_message: Message = {
                        "role": "assistant",
                        "content": assistant_message_blocks,
                    }
                    conversation.append(assistant_message)
                else:
                    logger.warning("Received empty assistant response from API")
                    return GenerationResult(
                        conversation=conversation, finish_reason="error"
                    )

                # Check if we need to execute tools
                if oai_message.tool_calls and args.tool_use_executor is not None:
                    # Execute all tool_use blocks and collect results
                    tool_result_blocks: list[ContentBlock] = []
                    for block in assistant_message_blocks:
                        if block["type"] == "tool_use":
                            try:
                                result_block = args.tool_use_executor(block)
                                tool_result_blocks.append(result_block)
                            except Exception as e:
                                logger.error(f"Tool execution error: {e}")
                                error_result: ToolResultBlock = {
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": f"Error: {str(e)}",
                                    "is_error": True,
                                }
                                tool_result_blocks.append(error_result)

                    # Add tool results to conversation and continue the loop
                    if tool_result_blocks:
                        conversation.append(
                            {"role": "user", "content": tool_result_blocks}
                        )
                        # Continue the loop to get model's response using the tool results
                        continue

                # If we reach here, either no tool calls or no executor
                # Map OpenAI finish_reason to our FinishReason type
                finish_reason_map: dict[str, FinishReason] = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "content_filter": "refusal",
                }
                finish_reason: FinishReason = finish_reason_map.get(
                    choice.finish_reason, "end_turn"
                )
                return GenerationResult(
                    conversation=conversation, finish_reason=finish_reason
                )

            # Max iterations reached
            logger.warning(f"Max tool use iterations ({max_tool_iterations}) reached")
            return GenerationResult(conversation=conversation, finish_reason="end_turn")

        except LengthFinishReasonError as e:
            logger.error(f"Max length error: {e}")
            return GenerationResult(
                conversation=conversation, finish_reason="max_tokens"
            )
        except Exception as e:
            logger.error(f"Error while generating: {e}")
            traceback.print_exc()
            return GenerationResult(conversation=conversation, finish_reason="error")


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
        self.model_name = args.model_id or "claude-sonnet-4-20250514"

    def _normalize_conversation(self, messages: list[Message]) -> list[Message]:
        """Normalize conversation for Anthropic API requirements.

        Anthropic API requires:
        1. Strictly alternating user/assistant messages
        2. Conversation must start with a user message

        This function:
        - Merges consecutive messages of the same role
        - Ensures the conversation starts with a user message
        """
        if not messages:
            return messages

        # First, merge consecutive messages of the same role
        merged: list[Message] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            if merged and merged[-1].get("role") == role:
                # Merge with previous message of same role
                existing_content = list(merged[-1].get("content", []))
                existing_content.extend(content)
                merged[-1] = {"role": role, "content": existing_content}  # type: ignore[typeddict-item]
            else:
                # Add as new message (copy to avoid mutation)
                merged.append({"role": role, "content": list(content)})  # type: ignore[typeddict-item]

        # Ensure conversation starts with user message
        if merged and merged[0].get("role") == "assistant":
            # Insert a placeholder user message at the start
            placeholder: Message = {
                "role": "user",
                "content": [{"type": "text", "text": "Continue."}],  # type: ignore[list-item]
            }
            merged.insert(0, placeholder)

        return merged

    async def generate(
        self, conversation: Conversation, args: GenerationArgs | None = None
    ) -> GenerationResult:
        args = args or self.gen_wrapper_args.default_generation_args
        max_tool_iterations = 10
        iteration = 0

        try:
            # Extract system message if present (Anthropic uses separate system param)
            system_content: str | None = None
            messages_for_api: list[Message] = []

            for msg in conversation:
                if msg["role"] == "system":
                    # Extract text from system message
                    content = msg.get("content", [])
                    if content and content[0].get("type") == "text":
                        system_content = content[0].get("text", "")
                else:
                    messages_for_api.append(msg)

            # Normalize conversation for Anthropic requirements (alternating roles, starts with user)
            messages_for_api = self._normalize_conversation(messages_for_api)
            log_conversation(messages_for_api)

            # Handle prefill by adding a partial assistant message
            if args.prefill:
                prefill_text_block: TextBlock = {"type": "text", "text": args.prefill}
                prefill_message: Message = {
                    "role": "assistant",
                    "content": [prefill_text_block],
                }
                messages_for_api.append(prefill_message)

            # Tool use loop: continue until we get a non-tool_use response
            while iteration < max_tool_iterations:
                iteration += 1

                # Build API request kwargs
                request_kwargs: dict = {
                    "model": self.model_name,
                    "max_tokens": args.max_tokens,
                    "messages": messages_for_api,
                }

                if system_content:
                    request_kwargs["system"] = system_content

                if args.temperature is not None:
                    request_kwargs["temperature"] = args.temperature

                if args.stop:
                    request_kwargs["stop_sequences"] = args.stop

                if args.tools:
                    request_kwargs["tools"] = args.tools
                    # Only set tool_choice on the first iteration - after that let model decide
                    if args.tool_choice and iteration == 1:
                        request_kwargs["tool_choice"] = args.tool_choice

                # Log request for debugging
                logger.debug(
                    f"Anthropic API request: model={request_kwargs.get('model')}, "
                    f"num_messages={len(request_kwargs.get('messages', []))}, "
                    f"message_roles={[m.get('role') for m in request_kwargs.get('messages', [])]}"
                )

                # Make the API call
                result = await self.client.messages.create(**request_kwargs)

                # Log raw response for debugging
                logger.debug(
                    f"Anthropic API response: stop_reason={result.stop_reason}, "
                    f"content_count={len(result.content)}, "
                    f"usage={result.usage}"
                )

                # Build content blocks from response
                assistant_message_blocks: list[ContentBlock] = []

                for block in result.content:
                    if block.type == "text":
                        text_content = block.text
                        # Prepend prefill to first text block for consistency
                        if args.prefill and iteration == 1:
                            text_content = args.prefill + text_content

                        text_block: TextBlock = {"type": "text", "text": text_content}
                        assistant_message_blocks.append(text_block)

                    elif block.type == "tool_use":
                        tool_use_block: ToolUseBlock = {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        assistant_message_blocks.append(tool_use_block)
                    else:
                        # Handle other block types (e.g., thinking blocks from extended thinking)
                        logger.debug(f"Skipping unhandled block type: {block.type}")

                # Add the assistant message to the conversation
                if assistant_message_blocks:
                    assistant_message: Message = {
                        "role": "assistant",
                        "content": assistant_message_blocks,
                    }
                    messages_for_api.append(assistant_message)
                    conversation.append(assistant_message)
                elif iteration > 1:
                    # Empty response after tool use is valid - model has nothing to add
                    # The previous tool_use response was the actual output
                    logger.debug(
                        f"Empty response after tool use (iteration {iteration}), "
                        f"treating as complete"
                    )
                    # Map Anthropic stop_reason to our FinishReason type
                    finish_reason_map: dict[str, FinishReason] = {
                        "end_turn": "end_turn",
                        "max_tokens": "max_tokens",
                        "stop_sequence": "stop_sequence",
                        "tool_use": "tool_use",
                    }
                    finish_reason: FinishReason = finish_reason_map.get(
                        result.stop_reason or "end_turn", "end_turn"
                    )
                    return GenerationResult(
                        conversation=conversation, finish_reason=finish_reason
                    )
                else:
                    # Empty response on first iteration is an error
                    logger.warning(
                        f"Received empty assistant response from Anthropic API. "
                        f"stop_reason={result.stop_reason}, "
                        f"content_types={[b.type for b in result.content]}, "
                        f"model={result.model}"
                    )
                    return GenerationResult(
                        conversation=conversation, finish_reason="error"
                    )

                # Check if we need to execute tools
                if (
                    result.stop_reason == "tool_use"
                    and args.tool_use_executor is not None
                ):
                    # Execute all tool_use blocks and collect results
                    tool_result_blocks: list[ContentBlock] = []

                    for block in assistant_message_blocks:
                        if block["type"] == "tool_use":
                            try:
                                result_block = args.tool_use_executor(block)  # type: ignore[arg-type]
                                tool_result_blocks.append(result_block)
                            except Exception as e:
                                logger.error(f"Tool execution error: {e}")
                                error_result: ToolResultBlock = {
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": f"Error: {str(e)}",
                                    "is_error": True,
                                }
                                tool_result_blocks.append(error_result)

                    # Add tool results as a user message and continue the loop
                    if tool_result_blocks:
                        tool_result_message: Message = {
                            "role": "user",
                            "content": tool_result_blocks,
                        }
                        messages_for_api.append(tool_result_message)
                        conversation.append(tool_result_message)
                        continue

                # Map Anthropic stop_reason to our FinishReason type
                finish_reason_map: dict[str, FinishReason] = {
                    "end_turn": "end_turn",
                    "max_tokens": "max_tokens",
                    "stop_sequence": "stop_sequence",
                    "tool_use": "tool_use",
                }
                finish_reason: FinishReason = finish_reason_map.get(
                    result.stop_reason or "end_turn", "end_turn"
                )
                return GenerationResult(
                    conversation=conversation, finish_reason=finish_reason
                )

            # Max iterations reached
            logger.warning(f"Max tool use iterations ({max_tool_iterations}) reached")
            return GenerationResult(conversation=conversation, finish_reason="end_turn")

        except Exception as e:
            logger.error(f"Error while generating with Anthropic: {e}")
            traceback.print_exc()
            return GenerationResult(conversation=conversation, finish_reason="error")


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
