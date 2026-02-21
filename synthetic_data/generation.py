"""Generation wrappers for provider-specific chat APIs.

Main flow:
1) Normalize conversations and generation args.
2) Apply token-rate/concurrency limits.
3) Call provider API, optionally execute tool calls.
4) Return unified `GenerationResult` records.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from anthropic import AsyncAnthropic
from anthropic.types.message import Message as AnthropicMessage
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from synthetic_data.generation_utils import (
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
    ToolResultBlock,
    ToolUseBlock,
    get_class_name,
)


def _merge_generation_args(
    default_args: GenWrapperArgs, override: GenerationArgs | None
) -> GenerationArgs:
    effective = default_args.default_generation_args.model_copy(deep=True)
    if override is None:
        return effective
    for key, value in override.model_dump(exclude_none=True).items():
        setattr(effective, key, value)
    return effective


def _normalize_conversations(
    conversation: Conversation | list[Conversation],
) -> tuple[list[Conversation], bool]:
    if not conversation:
        return [], True
    if isinstance(conversation[0], dict):
        return [conversation], False
    return conversation, True


def _parse_tool_input(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _content_blocks_to_text(blocks: list[ContentBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _map_openai_finish_reason(reason: str | None) -> str:
    if reason is None:
        return "end_turn"
    if reason == "stop":
        return "end_turn"
    if reason == "length":
        return "max_tokens"
    if reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if reason == "content_filter":
        return "refusal"
    return reason


class TokenBucketRateLimiter:
    """Token-per-second limiter with adaptive request-size estimation.

    The caller acquires an estimated token budget pre-request, then reconciles
    with actual usage by debiting or refunding the bucket.
    """

    def __init__(
        self,
        max_tokens_per_sec: float,
        initial_tokens_per_request: float = 1024.0,
        alpha: float = 0.2,
    ) -> None:
        self.max_tps = max(0.0, float(max_tokens_per_sec))
        self.tokens_per_request = max(1.0, float(initial_tokens_per_request))
        self.alpha = alpha
        self.capacity = max(self.max_tps, self.tokens_per_request)
        self.tokens = self.capacity
        self.last_ts = time.monotonic()
        self._lock = asyncio.Lock()

    def estimate_tokens(self, n_items: int = 1) -> float:
        return max(1.0, self.tokens_per_request * max(1, n_items))

    async def acquire(self, est_tokens: float) -> None:
        if self.max_tps <= 0:
            return
        while True:
            async with self._lock:
                self._refill_locked()
                if self.tokens >= est_tokens:
                    self.tokens -= est_tokens
                    return
                deficit = est_tokens - self.tokens
                wait_time = deficit / self.max_tps if self.max_tps > 0 else 0.05
            await asyncio.sleep(max(wait_time, 0.0))

    async def refund(self, refunded_tokens: float) -> None:
        if refunded_tokens <= 0:
            return
        async with self._lock:
            self._refill_locked()
            self.tokens = min(self.capacity, self.tokens + refunded_tokens)

    async def debit(self, extra_tokens: float) -> None:
        if extra_tokens <= 0:
            return
        async with self._lock:
            self._refill_locked()
            # Allow temporary debt; future acquires wait for refill.
            self.tokens -= extra_tokens

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_ts
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.max_tps)
            self.last_ts = now

    def update(self, actual_tokens: int, n_items: int = 1) -> None:
        if actual_tokens <= 0:
            return
        per_request = max(1.0, actual_tokens / max(1, n_items))
        self.tokens_per_request = (
            1 - self.alpha
        ) * self.tokens_per_request + self.alpha * per_request
        self.capacity = max(self.max_tps, self.tokens_per_request)


class ThroughputConcurrencyLimiter:
    """Limit in-flight requests based on the model token throughput."""

    def __init__(self, rate_limiter: TokenBucketRateLimiter) -> None:
        self._rate_limiter = rate_limiter
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)

    def _max_in_flight(self) -> int:
        max_tps = self._rate_limiter.max_tps
        if max_tps <= 0:
            return 1_000_000
        est_tokens = max(1.0, self._rate_limiter.tokens_per_request)
        return max(1, int(max_tps // est_tokens))

    async def acquire(self) -> None:
        async with self._cond:
            while self._in_flight >= self._max_in_flight():
                await self._cond.wait()
            self._in_flight += 1

    async def release(self) -> None:
        async with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            self._cond.notify_all()

    async def __aenter__(self) -> "ThroughputConcurrencyLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class OpenAIGenerationWrapper(GenerationWrapper):
    """OpenAI Chat Completions wrapper with tool-calling support."""

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
        initial_estimate = (
            default_args.default_generation_args.max_tokens
            if default_args.default_generation_args.max_tokens
            else 1024
        )
        self._rate_limiter = TokenBucketRateLimiter(
            default_args.max_rps, initial_tokens_per_request=initial_estimate
        )
        self._concurrency_limiter = ThroughputConcurrencyLimiter(self._rate_limiter)

    async def generate(
        self,
        conversation: Conversation | list[Conversation],
        args: GenerationArgs | None = None,
    ) -> GenerationResult | list[GenerationResult]:
        """Generate one or many conversations and return normalized results."""
        conversations, is_batch = _normalize_conversations(conversation)
        effective_args = _merge_generation_args(self.gen_wrapper_args, args)

        async def _execute_tool_use(
            executor: Any, tool_use: ToolUseBlock
        ) -> ToolResultBlock:
            result = executor(tool_use)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        def _to_openai_messages(conv: Conversation) -> list[dict]:
            messages: list[dict] = []
            for msg in conv:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    tool_use_blocks = [
                        block for block in content if block.get("type") == "tool_use"
                    ]
                    tool_result_blocks = [
                        block for block in content if block.get("type") == "tool_result"
                    ]
                    text_blocks = [
                        block for block in content if block.get("type") == "text"
                    ]

                    if tool_result_blocks:
                        for block in tool_result_blocks:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(block.get("content", "")),
                                }
                            )
                        if text_blocks:
                            messages.append(
                                {
                                    "role": "user" if role == "tool" else role,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": block.get("text", ""),
                                        }
                                        for block in text_blocks
                                        if block.get("text")
                                    ],
                                }
                            )
                        continue

                    if tool_use_blocks and role == "assistant":
                        tool_calls = []
                        for block in tool_use_blocks:
                            tool_input = block.get("input", {})
                            if not isinstance(tool_input, str):
                                tool_input = json.dumps(tool_input, ensure_ascii=False)
                            tool_calls.append(
                                {
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": tool_input,
                                    },
                                }
                            )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": block.get("text", ""),
                                    }
                                    for block in text_blocks
                                    if block.get("text")
                                ]
                                or None,
                                "tool_calls": tool_calls,
                            }
                        )
                        continue

                    messages.append(
                        {
                            "role": role,
                            "content": (
                                [
                                    {
                                        "type": "text",
                                        "text": block.get("text", ""),
                                    }
                                    for block in text_blocks
                                    if block.get("text")
                                ]
                                or ""
                            ),
                        }
                    )
                    continue

                messages.append({"role": role, "content": content})
            return messages

        def _to_tool_blocks_from_openai(tool_calls: Any) -> list[ToolUseBlock]:
            tool_blocks: list[ToolUseBlock] = []
            for tool_call in tool_calls:
                tool_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": _parse_tool_input(tool_call.function.arguments),
                    }
                )
            return tool_blocks

        def _build_tools_param(tools: list[dict]) -> list[dict]:
            converted = []
            for tool in tools:
                name = tool.get("name", "")
                description = tool.get("description", "")
                input_schema = tool.get("input_schema", {"type": "object"})
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": input_schema,
                        },
                    }
                )
            return converted

        def _map_tool_choice(choice: Any) -> Any:
            if choice is None:
                return None
            if choice == "auto" or choice == "none" or choice == "required":
                return choice
            if isinstance(choice, dict):
                if choice.get("type") == "any":
                    return "auto"
                if choice.get("type") == "tool" and "name" in choice:
                    return {
                        "type": "function",
                        "function": {"name": choice["name"]},
                    }
            return choice

        async def _call_openai(
            messages: list[dict],
            tools: list[dict] | None,
            tool_choice: Any | None,
        ) -> Any:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            if effective_args.max_tokens is not None:
                kwargs["max_tokens"] = effective_args.max_tokens
            if effective_args.temperature is not None:
                kwargs["temperature"] = effective_args.temperature
            if effective_args.seed is not None:
                kwargs["seed"] = effective_args.seed
            if tools:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body

            timeout_s = self.gen_wrapper_args.request_timeout_s
            if isinstance(self.oai_client, AsyncOpenAI):
                return await asyncio.wait_for(
                    self.oai_client.chat.completions.create(**kwargs),
                    timeout=timeout_s,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(self.oai_client.chat.completions.create, **kwargs),
                timeout=timeout_s,
            )

        async def _generate_one(conv: Conversation) -> GenerationResult:
            messages = _to_openai_messages(conv)
            tools_param = (
                _build_tools_param(effective_args.tools)
                if effective_args.tools
                else None
            )
            tool_choice_param = _map_tool_choice(effective_args.tool_choice)
            added_messages: list[Message] = []
            total_usage = 0

            n_retries = max(1, effective_args.n_retries)
            for attempt in range(n_retries):
                try:
                    tool_iterations = 0
                    while True:
                        # Reserve estimated tokens before issuing the request.
                        est_tokens = self._rate_limiter.estimate_tokens()
                        await self._rate_limiter.acquire(est_tokens)
                        async with self._concurrency_limiter:
                            response = await _call_openai(
                                messages, tools_param, tool_choice_param
                            )
                        if response.usage and response.usage.total_tokens:
                            # Reconcile estimate with actual token usage.
                            actual_tokens = response.usage.total_tokens
                            total_usage += actual_tokens
                            if actual_tokens > est_tokens:
                                await self._rate_limiter.debit(
                                    actual_tokens - est_tokens
                                )
                            elif est_tokens > actual_tokens:
                                await self._rate_limiter.refund(
                                    est_tokens - actual_tokens
                                )
                            self._rate_limiter.update(actual_tokens)

                        choice = response.choices[0]
                        message = choice.message
                        tool_calls = message.tool_calls or []
                        content_blocks: list[ContentBlock] = []
                        if message.content:
                            content_blocks.append(
                                {"type": "text", "text": message.content}
                            )
                        if tool_calls:
                            content_blocks.extend(
                                _to_tool_blocks_from_openai(tool_calls)
                            )

                        assistant_message: Message = {
                            "role": "assistant",
                            "content": content_blocks,
                        }
                        added_messages.append(assistant_message)
                        assistant_payload: dict = {
                            "role": "assistant",
                            "content": message.content,
                        }
                        if tool_calls:
                            assistant_payload["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ]
                        messages.append(assistant_payload)

                        if tool_calls and effective_args.tool_use_executor:
                            # Continue the same turn by feeding tool results back.
                            tool_iterations += 1
                            if tool_iterations > effective_args.max_tool_iterations:
                                return GenerationResult(
                                    added_messages=added_messages,
                                    finish_reason="error",
                                    content=None,
                                    conversation=conv + added_messages,
                                    usage={"total_tokens": total_usage}
                                    if total_usage
                                    else None,
                                )
                            tool_result_blocks = []
                            tool_messages = []
                            for tool_use in _to_tool_blocks_from_openai(tool_calls):
                                result_block = await _execute_tool_use(
                                    effective_args.tool_use_executor, tool_use
                                )
                                tool_result_blocks.append(result_block)
                                tool_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": result_block.get(
                                            "tool_use_id", ""
                                        ),
                                        "content": str(result_block.get("content", "")),
                                    }
                                )
                            added_messages.append(
                                {"role": "user", "content": tool_result_blocks}
                            )
                            messages.extend(tool_messages)
                            if tool_choice_param is not None:
                                tool_choice_param = "auto"
                            continue

                        final_content = message.content or _content_blocks_to_text(
                            content_blocks
                        )
                        return GenerationResult(
                            added_messages=added_messages,
                            finish_reason=_map_openai_finish_reason(
                                choice.finish_reason
                            ),
                            content=final_content,
                            conversation=conv + added_messages,
                            usage={"total_tokens": total_usage}
                            if total_usage
                            else None,
                        )
                except Exception as e:
                    if attempt == n_retries - 1:
                        logger.error(f"OpenAI generation failed: {e}")
                        return GenerationResult(
                            added_messages=added_messages,
                            finish_reason="error",
                            content=None,
                            conversation=conv + added_messages,
                            usage={"total_tokens": total_usage}
                            if total_usage
                            else None,
                        )
                    await asyncio.sleep(min(2**attempt, 10))

            return GenerationResult(
                added_messages=added_messages,
                finish_reason="error",
                content=None,
                conversation=conv + added_messages,
                usage={"total_tokens": total_usage} if total_usage else None,
            )

        results = await asyncio.gather(*[_generate_one(conv) for conv in conversations])
        return results if is_batch else results[0]


class VLLMWrapper(OpenAIGenerationWrapper):
    """OpenAI-compatible wrapper pointed at a local vLLM server."""

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
        self.n_retries = 10
        self.temperature = 0.2
        self.max_tokens = 4096
        self.args = args
        if args.model_id is None:
            logger.info("Fetching models from OpenAI client")
            sync_client = OpenAI(base_url=base_url)
            all_models = sync_client.models.list()
            logger.info(f"Models: {[x.id for x in all_models.data]}")
            self.model_name = all_models.data[0].id
            logger.info(f"Using model: {self.model_name}")


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):
    """OpenAI-compatible wrapper configured for OpenRouter."""

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
        if args.providers:
            logger.info(f"Using providers: {args.providers}")
            self.extra_body["provider"] = {"order": args.providers}


class AnthropicGenerationWrapper(GenerationWrapper):
    """Anthropic Messages API wrapper with normalized result conversion."""

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
        initial_estimate = (
            args.default_generation_args.max_tokens
            if args.default_generation_args.max_tokens
            else 1024
        )
        self._rate_limiter = TokenBucketRateLimiter(
            args.max_rps, initial_tokens_per_request=initial_estimate
        )
        self._concurrency_limiter = ThroughputConcurrencyLimiter(self._rate_limiter)

    async def generate(
        self,
        conversation: Conversation | list[Conversation],
        args: GenerationArgs | None = None,
    ) -> GenerationResult | list[GenerationResult]:
        """Generate one or many conversations and return normalized results."""
        conversations, is_batch = _normalize_conversations(conversation)
        effective_args = _merge_generation_args(self.gen_wrapper_args, args)

        async def _execute_tool_use(
            executor: Any, tool_use: ToolUseBlock
        ) -> ToolResultBlock:
            result = executor(tool_use)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        def _split_system(
            conv: Conversation,
        ) -> tuple[list[ContentBlock], list[Message]]:
            system_blocks: list[ContentBlock] = []
            messages: list[Message] = []
            for msg in conv:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                system_blocks.append(block)
                    elif content:
                        system_blocks.append({"type": "text", "text": content})
                    continue
                messages.append(msg)
            return system_blocks, messages

        def _normalize_anthropic_messages(msgs: list[Message]) -> list[dict]:
            normalized = []
            for msg in msgs:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    normalized.append({"role": role, "content": content})
                else:
                    normalized.append({"role": role, "content": content})
            return normalized

        async def _generate_one(conv: Conversation) -> GenerationResult:
            system_blocks, message_list = _split_system(conv)
            messages = _normalize_anthropic_messages(message_list)
            tools_param = effective_args.tools or None
            tool_choice_param = effective_args.tool_choice
            added_messages: list[Message] = []
            total_usage = 0

            n_retries = max(1, effective_args.n_retries)
            for attempt in range(n_retries):
                try:
                    tool_iterations = 0
                    while True:
                        # Reserve estimated tokens before issuing the request.
                        est_tokens = self._rate_limiter.estimate_tokens()
                        await self._rate_limiter.acquire(est_tokens)
                        request_kwargs = {
                            "model": self.model_name,
                            "max_tokens": effective_args.max_tokens or 1024,
                            "system": system_blocks if system_blocks else None,
                            "messages": messages,
                        }
                        # Thinking is incompatible with forced tool choice
                        # (tool_choice "any" or {"type": "tool", ...}).
                        forced_tool = False
                        if tool_choice_param is not None:
                            if isinstance(tool_choice_param, dict):
                                tc_type = tool_choice_param.get("type")
                                forced_tool = tc_type in ("any", "tool")
                            elif isinstance(tool_choice_param, str):
                                forced_tool = tool_choice_param in ("any", "tool")
                        if (
                            effective_args.thinking_budget_tokens is not None
                            and not forced_tool
                        ):
                            request_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": effective_args.thinking_budget_tokens,
                            }
                            # Temperature must be 1 (default) when thinking is enabled.
                        elif effective_args.temperature is not None:
                            request_kwargs["temperature"] = effective_args.temperature
                        if tools_param:
                            request_kwargs["tools"] = tools_param
                        if tool_choice_param is not None:
                            request_kwargs["tool_choice"] = tool_choice_param
                        async with self._concurrency_limiter:
                            response: AnthropicMessage = await asyncio.wait_for(
                                self.client.messages.create(**request_kwargs),
                                timeout=self.gen_wrapper_args.request_timeout_s,
                            )

                        if response.usage:
                            usage_tokens = 0
                            for key in [
                                "input_tokens",
                                "output_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                            ]:
                                usage_tokens += int(getattr(response.usage, key, 0) or 0)
                            if usage_tokens:
                                # Reconcile estimate with actual token usage.
                                total_usage += usage_tokens
                                if usage_tokens > est_tokens:
                                    await self._rate_limiter.debit(
                                        usage_tokens - est_tokens
                                    )
                                elif est_tokens > usage_tokens:
                                    await self._rate_limiter.refund(
                                        est_tokens - usage_tokens
                                    )
                                self._rate_limiter.update(usage_tokens)

                        assistant_blocks: list[ContentBlock] = []
                        for block in response.content:
                            if block.type == "thinking":
                                assistant_blocks.append(
                                    {"type": "thinking", "thinking": block.thinking}
                                )
                            elif block.type == "text":
                                assistant_blocks.append(
                                    {"type": "text", "text": block.text}
                                )
                            elif block.type == "tool_use":
                                assistant_blocks.append(
                                    {
                                        "type": "tool_use",
                                        "id": block.id,
                                        "name": block.name,
                                        "input": block.input,
                                    }
                                )
                        assistant_message: Message = {
                            "role": "assistant",
                            "content": assistant_blocks,
                        }
                        added_messages.append(assistant_message)
                        messages.append(assistant_message)

                        tool_use_blocks = [
                            block
                            for block in assistant_blocks
                            if block.get("type") == "tool_use"
                        ]
                        if (
                            response.stop_reason == "tool_use"
                            and tool_use_blocks
                            and effective_args.tool_use_executor
                        ):
                            # Continue the same turn by feeding tool results back.
                            tool_iterations += 1
                            if tool_iterations > effective_args.max_tool_iterations:
                                return GenerationResult(
                                    added_messages=added_messages,
                                    finish_reason="error",
                                    content=None,
                                    conversation=conv + added_messages,
                                    usage={
                                        "total_tokens": total_usage,
                                        "input_tokens": response.usage.input_tokens,
                                        "output_tokens": response.usage.output_tokens,
                                    }
                                    if response.usage
                                    else None,
                                )
                            tool_results: list[ToolResultBlock] = []
                            for tool_use in tool_use_blocks:
                                result_block = await _execute_tool_use(
                                    effective_args.tool_use_executor, tool_use
                                )
                                tool_results.append(result_block)
                            tool_result_message: Message = {
                                "role": "user",
                                "content": tool_results,
                            }
                            added_messages.append(tool_result_message)
                            messages.append(tool_result_message)
                            tool_choice_param = None
                            continue

                        final_content = _content_blocks_to_text(assistant_blocks)
                        return GenerationResult(
                            added_messages=added_messages,
                            finish_reason=response.stop_reason or "end_turn",
                            content=final_content,
                            conversation=conv + added_messages,
                            usage={"total_tokens": total_usage}
                            if total_usage
                            else None,
                        )
                except Exception as e:
                    if attempt == n_retries - 1:
                        logger.error(f"Anthropic generation failed: {e}")
                        return GenerationResult(
                            added_messages=added_messages,
                            finish_reason="error",
                            content=None,
                            conversation=conv + added_messages,
                            usage={
                                "total_tokens": total_usage,
                                "input_tokens": response.usage.input_tokens,
                                "output_tokens": response.usage.output_tokens,
                            }
                            if response.usage
                            else None,
                        )
                    await asyncio.sleep(min(2**attempt, 10))

            return GenerationResult(
                added_messages=added_messages,
                finish_reason="error",
                content=None,
                conversation=conv + added_messages,
                usage={
                    "total_tokens": total_usage,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if response.usage
                else None,
            )

        results = await asyncio.gather(*[_generate_one(conv) for conv in conversations])
        return results if is_batch else results[0]


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
    """Build a wrapper from `MODEL_CONFIGS`, applying optional arg overrides."""
    config = MODEL_CONFIGS[model_name]
    base_args = (
        config.args.model_copy(deep=True)
        if config.args is not None
        else GenWrapperArgs()
    )
    if args_override is not None:
        for key, value in args_override.model_dump(exclude_none=True).items():
            setattr(base_args, key, value)
            logger.info(f"Overriding {key} in gen wrapper args with {value}")
    return config.model(base_args)
