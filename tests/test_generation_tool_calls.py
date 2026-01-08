"""Integration tests for generation wrappers with tool call processing."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synthetic_data.generation import (
    AnthropicGenerationWrapper,
    OpenAIGenerationWrapper,
)
from synthetic_data.generation_utils import GenerationArgs, GenWrapperArgs
from synthetic_data.utils import (
    Conversation,
    ToolResultBlock,
    ToolUseBlock,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_tool_executor():
    """Create a tool executor that tracks calls and returns predictable results."""
    calls = []

    def executor(tool_use: ToolUseBlock) -> ToolResultBlock:
        calls.append(tool_use)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": f"Result for {tool_use['name']}",
        }

    executor.calls = calls  # type: ignore
    return executor


@pytest.fixture
def mock_openai_env():
    """Set up mock environment for OpenAI."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_anthropic_env():
    """Set up mock environment for Anthropic."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        yield


def create_openai_tool_call_response(tool_id: str, tool_name: str, arguments: str):
    """Create a mock OpenAI ChatCompletion with tool calls."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = tool_id
    mock_tool_call.type = "function"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = tool_name
    mock_tool_call.function.arguments = arguments

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    return mock_completion


def create_openai_text_response(text: str, finish_reason: str = "stop"):
    """Create a mock OpenAI ChatCompletion with text content."""
    mock_message = MagicMock()
    mock_message.content = text
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    return mock_completion


def create_anthropic_tool_use_response(tool_id: str, tool_name: str, tool_input: dict):
    """Create a mock Anthropic Message with tool use."""
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = tool_id
    mock_tool_block.name = tool_name
    mock_tool_block.input = tool_input

    mock_message = MagicMock()
    mock_message.content = [mock_tool_block]
    mock_message.stop_reason = "tool_use"
    mock_message.usage = MagicMock()

    return mock_message


def create_anthropic_text_response(text: str, stop_reason: str = "end_turn"):
    """Create a mock Anthropic Message with text content."""
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = text

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.stop_reason = stop_reason
    mock_message.usage = MagicMock()

    return mock_message


def test_openai_wrapper_executes_tool_and_continues(mock_openai_env):
    """Test that OpenAI wrapper executes tools and continues the generation loop."""
    tool_executor = make_tool_executor()

    # Mock responses: first returns tool call, second returns final text
    tool_response = create_openai_tool_call_response(
        tool_id="call_123",
        tool_name="get_weather",
        arguments='{"city": "San Francisco"}',
    )
    final_response = create_openai_text_response("The weather is sunny!")

    with patch("synthetic_data.generation.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        mock_client_class.return_value = mock_client

        wrapper = OpenAIGenerationWrapper(
            GenWrapperArgs(model_id="gpt-4o-mini", use_async_client=True)
        )

        conversation: Conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather?"}],
            }
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]

        result = run_async(
            wrapper.generate(
                conversation,
                GenerationArgs(tools=tools, tool_use_executor=tool_executor),
            )
        )

        # Verify tool executor was called
        assert len(tool_executor.calls) == 1
        assert tool_executor.calls[0]["name"] == "get_weather"
        assert tool_executor.calls[0]["id"] == "call_123"

        # Verify conversation structure
        assert (
            len(result.conversation) == 4
        )  # user, assistant+tool_use, user+tool_result, assistant
        assert result.conversation[0].get("role") == "user"
        assert result.conversation[1].get("role") == "assistant"
        assert result.conversation[2].get("role") == "user"
        assert result.conversation[3].get("role") == "assistant"

        # Verify tool_use block in assistant message
        assistant_content = result.conversation[1].get("content", [])
        tool_use_blocks = [b for b in assistant_content if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"

        # Verify tool_result block in user message
        tool_result_content = result.conversation[2].get("content", [])
        assert tool_result_content[0].get("type") == "tool_result"
        assert tool_result_content[0].get("tool_use_id") == "call_123"

        # Verify final response
        assert result.finish_reason == "end_turn"
        final_content = result.conversation[3].get("content", [])
        assert final_content[0].get("text") == "The weather is sunny!"


def test_openai_wrapper_handles_tool_executor_error(mock_openai_env):
    """Test that OpenAI wrapper handles tool executor errors gracefully."""

    def failing_executor(tool_use: ToolUseBlock) -> ToolResultBlock:
        raise ValueError("Tool execution failed!")

    tool_response = create_openai_tool_call_response(
        tool_id="call_456",
        tool_name="broken_tool",
        arguments="{}",
    )
    final_response = create_openai_text_response("I see there was an error.")

    with patch("synthetic_data.generation.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        mock_client_class.return_value = mock_client

        wrapper = OpenAIGenerationWrapper(
            GenWrapperArgs(model_id="gpt-4o-mini", use_async_client=True)
        )

        conversation: Conversation = [
            {"role": "user", "content": [{"type": "text", "text": "Run the tool"}]}
        ]

        result = run_async(
            wrapper.generate(
                conversation,
                GenerationArgs(
                    tools=[
                        {
                            "name": "broken_tool",
                            "description": "A broken tool",
                            "input_schema": {},
                        }
                    ],
                    tool_use_executor=failing_executor,
                ),
            )
        )

        # Verify error was captured in tool result
        tool_result_msg = result.conversation[2]
        assert tool_result_msg.get("role") == "user"
        tool_result = tool_result_msg.get("content", [])[0]
        assert tool_result.get("type") == "tool_result"
        assert tool_result.get("is_error") is True
        assert "Tool execution failed!" in tool_result.get("content", "")


def test_anthropic_wrapper_executes_tool_and_continues(mock_anthropic_env):
    """Test that Anthropic wrapper executes tools and continues the generation loop."""
    tool_executor = make_tool_executor()

    tool_response = create_anthropic_tool_use_response(
        tool_id="toolu_789",
        tool_name="calculate",
        tool_input={"expression": "2 + 2"},
    )
    final_response = create_anthropic_text_response("The result is 4.")

    with patch("synthetic_data.generation.AsyncAnthropic") as mock_client_class:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        mock_client_class.return_value = mock_client

        wrapper = AnthropicGenerationWrapper(
            GenWrapperArgs(model_id="claude-sonnet-4-20250514")
        )

        conversation: Conversation = [
            {"role": "user", "content": [{"type": "text", "text": "Calculate 2 + 2"}]}
        ]

        tools = [
            {
                "name": "calculate",
                "description": "Evaluate a math expression",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            }
        ]

        result = run_async(
            wrapper.generate(
                conversation,
                GenerationArgs(tools=tools, tool_use_executor=tool_executor),
            )
        )

        # Verify tool executor was called
        assert len(tool_executor.calls) == 1
        assert tool_executor.calls[0]["name"] == "calculate"
        assert tool_executor.calls[0]["id"] == "toolu_789"

        # Verify conversation structure
        assert len(result.conversation) == 4
        assert result.conversation[1].get("role") == "assistant"
        assert result.conversation[2].get("role") == "user"
        assert result.conversation[3].get("role") == "assistant"

        # Verify tool_use block
        assistant_content = result.conversation[1].get("content", [])
        assert assistant_content[0]["type"] == "tool_use"
        assert assistant_content[0]["name"] == "calculate"

        # Verify tool_result block
        tool_result_content = result.conversation[2].get("content", [])
        assert tool_result_content[0].get("type") == "tool_result"
        assert tool_result_content[0].get("tool_use_id") == "toolu_789"

        # Verify finish reason
        assert result.finish_reason == "end_turn"


class TestOpenAIConversationConversion:
    """Test OpenAI conversation format conversion."""

    def test_convert_tool_use_blocks_to_openai_format(self, mock_openai_env):
        """Test that internal tool_use blocks convert correctly to OpenAI format."""
        with patch("synthetic_data.generation.AsyncOpenAI"):
            wrapper = OpenAIGenerationWrapper(
                GenWrapperArgs(model_id="gpt-4o-mini", use_async_client=True)
            )

            conversation: Conversation = [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_info",
                            "input": {"query": "test"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Result data",
                        }
                    ],
                },
            ]

            openai_messages = wrapper._convert_conversation_to_openai_format(
                conversation
            )

            # Verify user message
            assert openai_messages[0]["role"] == "user"
            assert openai_messages[0]["content"] == "Hello"

            # Verify assistant message with tool_calls
            assert openai_messages[1]["role"] == "assistant"
            assert "tool_calls" in openai_messages[1]
            assert openai_messages[1]["tool_calls"][0]["id"] == "call_123"
            assert openai_messages[1]["tool_calls"][0]["function"]["name"] == "get_info"

            # Verify tool result message
            assert openai_messages[2]["role"] == "tool"
            assert openai_messages[2]["tool_call_id"] == "call_123"
            assert openai_messages[2]["content"] == "Result data"

    def test_convert_tools_to_openai_format(self, mock_openai_env):
        """Test that Anthropic-style tools convert correctly to OpenAI format."""
        with patch("synthetic_data.generation.AsyncOpenAI"):
            wrapper = OpenAIGenerationWrapper(
                GenWrapperArgs(model_id="gpt-4o-mini", use_async_client=True)
            )

            anthropic_tools = [
                {
                    "name": "roll_dice",
                    "description": "Roll dice using RPG notation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "notation": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["notation", "reason"],
                    },
                }
            ]

            openai_tools = wrapper._convert_tools_to_openai_format(anthropic_tools)

            assert len(openai_tools) == 1
            assert openai_tools[0]["type"] == "function"
            assert openai_tools[0]["function"]["name"] == "roll_dice"
            assert (
                openai_tools[0]["function"]["description"]
                == "Roll dice using RPG notation"
            )
            assert "properties" in openai_tools[0]["function"]["parameters"]
