"""Unit tests for the roleplaying game task and tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ToolCallFunction,
)

from synthetic_data.generation import GenerationResult
from synthetic_data.tasks.roleplaying_tools import (
    ROLL_DICE_TOOL,
    RANDOM_CHOICE_TOOL,
    PRESENT_CHOICES_TOOL,
    SPEAK_TOOL,
    ACTION_TOOL,
    DM_TOOLS,
    PLAYER_TOOLS,
    ALL_TOOLS,
    ToolResult,
    convert_to_openai_format,
    parse_dice_notation,
    execute_roll_dice,
    execute_random_choice,
    execute_present_choices,
    execute_speak,
    execute_action,
    execute_tool_call,
    execute_tool_calls,
)
from synthetic_data.tasks.roleplaying import (
    format_tool_result_for_conversation,
    format_tool_calls_for_conversation,
    generate_with_tools_loop,
    RPGEpisode,
    RoleplayingGameMultiStepTask,
)


def make_tool_call(id: str, name: str, arguments: dict) -> ChatCompletionMessageToolCall:
    """Helper to create a ChatCompletionMessageToolCall."""
    return ChatCompletionMessageToolCall(
        id=id,
        type="function",
        function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )


class TestToolSchemas:
    """Test that tool schemas are properly defined."""

    def test_roll_dice_schema(self):
        assert ROLL_DICE_TOOL["name"] == "roll_dice"
        assert "description" in ROLL_DICE_TOOL
        assert "input_schema" in ROLL_DICE_TOOL
        schema = ROLL_DICE_TOOL["input_schema"]
        assert schema["type"] == "object"
        assert "notation" in schema["properties"]
        assert "reason" in schema["properties"]
        assert "notation" in schema["required"]
        assert "reason" in schema["required"]

    def test_random_choice_schema(self):
        assert RANDOM_CHOICE_TOOL["name"] == "random_choice"
        schema = RANDOM_CHOICE_TOOL["input_schema"]
        assert "options" in schema["properties"]
        assert schema["properties"]["options"]["type"] == "array"

    def test_present_choices_schema(self):
        assert PRESENT_CHOICES_TOOL["name"] == "present_choices"
        schema = PRESENT_CHOICES_TOOL["input_schema"]
        assert "prompt" in schema["properties"]
        assert "choices" in schema["properties"]

    def test_speak_schema(self):
        assert SPEAK_TOOL["name"] == "speak"
        schema = SPEAK_TOOL["input_schema"]
        assert "character" in schema["properties"]
        assert "message" in schema["properties"]
        assert "tone" in schema["properties"]
        assert "character" in schema["required"]
        assert "message" in schema["required"]
        assert "tone" not in schema["required"]

    def test_action_schema(self):
        assert ACTION_TOOL["name"] == "action"
        schema = ACTION_TOOL["input_schema"]
        assert "description" in schema["properties"]
        assert "target" in schema["properties"]
        assert "description" in schema["required"]
        assert "target" not in schema["required"]

    def test_tool_collections(self):
        assert len(DM_TOOLS) == 4
        assert ROLL_DICE_TOOL in DM_TOOLS
        assert RANDOM_CHOICE_TOOL in DM_TOOLS
        assert PRESENT_CHOICES_TOOL in DM_TOOLS
        assert SPEAK_TOOL in DM_TOOLS

        assert len(PLAYER_TOOLS) == 3
        assert ROLL_DICE_TOOL in PLAYER_TOOLS
        assert SPEAK_TOOL in PLAYER_TOOLS
        assert ACTION_TOOL in PLAYER_TOOLS

        assert len(ALL_TOOLS) == 5


class TestOpenAIFormatConversion:
    """Test conversion from Anthropic to OpenAI tool format."""

    def test_convert_single_tool(self):
        result = convert_to_openai_format([ROLL_DICE_TOOL])
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "roll_dice"
        assert tool["function"]["description"] == ROLL_DICE_TOOL["description"]
        assert tool["function"]["parameters"] == ROLL_DICE_TOOL["input_schema"]

    def test_convert_multiple_tools(self):
        result = convert_to_openai_format(DM_TOOLS)
        assert len(result) == 4
        names = [t["function"]["name"] for t in result]
        assert "roll_dice" in names
        assert "random_choice" in names
        assert "present_choices" in names
        assert "speak" in names


class TestDiceNotationParsing:
    """Test dice notation parsing."""

    def test_simple_notation(self):
        result = parse_dice_notation("1d20")
        assert result == (1, 20, 0)

    def test_multiple_dice(self):
        result = parse_dice_notation("3d6")
        assert result == (3, 6, 0)

    def test_positive_modifier(self):
        result = parse_dice_notation("2d6+3")
        assert result == (2, 6, 3)

    def test_negative_modifier(self):
        result = parse_dice_notation("1d8-2")
        assert result == (1, 8, -2)

    def test_implicit_count(self):
        result = parse_dice_notation("d20")
        assert result == (1, 20, 0)

    def test_case_insensitive(self):
        result = parse_dice_notation("2D10+5")
        assert result == (2, 10, 5)

    def test_invalid_notation(self):
        result = parse_dice_notation("invalid")
        assert result is None

    def test_empty_string(self):
        result = parse_dice_notation("")
        assert result is None


class TestToolExecution:
    """Test tool execution functions."""

    def test_execute_roll_dice_valid(self):
        result = execute_roll_dice("2d6", "attack roll")
        assert result.success
        assert result.content["notation"] == "2d6"
        assert result.content["reason"] == "attack roll"
        assert len(result.content["rolls"]) == 2
        assert all(1 <= r <= 6 for r in result.content["rolls"])
        assert result.content["total"] == sum(result.content["rolls"])

    def test_execute_roll_dice_with_modifier(self):
        result = execute_roll_dice("1d20+5", "skill check")
        assert result.success
        assert result.content["modifier"] == 5
        assert result.content["total"] == result.content["rolls"][0] + 5

    def test_execute_roll_dice_invalid(self):
        result = execute_roll_dice("invalid", "test")
        assert not result.success
        assert result.error is not None

    def test_execute_random_choice(self):
        options = ["left", "right", "straight"]
        result = execute_random_choice(options, "path selection")
        assert result.success
        assert result.content["chosen"] in options
        assert result.content["options"] == options
        assert result.content["reason"] == "path selection"

    def test_execute_random_choice_empty(self):
        result = execute_random_choice([], "empty test")
        assert not result.success
        assert result.error is not None

    def test_execute_present_choices(self):
        choices = [
            {"id": "a", "description": "Go left"},
            {"id": "b", "description": "Go right"},
        ]
        result = execute_present_choices("Which way?", choices)
        assert result.success
        assert result.content["prompt"] == "Which way?"
        assert result.content["choices"] == choices
        assert result.content["awaiting_player_choice"] is True

    def test_execute_speak(self):
        result = execute_speak("Gandalf", "You shall not pass!", "shouted")
        assert result.success
        assert result.content["character"] == "Gandalf"
        assert result.content["message"] == "You shall not pass!"
        assert result.content["tone"] == "shouted"

    def test_execute_speak_no_tone(self):
        result = execute_speak("Frodo", "I will take the ring.")
        assert result.success
        assert result.content["character"] == "Frodo"
        assert result.content["message"] == "I will take the ring."
        assert "tone" not in result.content

    def test_execute_action(self):
        result = execute_action("swing sword", "goblin")
        assert result.success
        assert result.content["description"] == "swing sword"
        assert result.content["target"] == "goblin"
        assert result.content["executed"] is True

    def test_execute_action_no_target(self):
        result = execute_action("look around")
        assert result.success
        assert result.content["description"] == "look around"
        assert "target" not in result.content


class TestToolCallExecution:
    """Test the tool call dispatch system."""

    def test_execute_tool_call_roll_dice(self):
        tool_call = make_tool_call(
            id="test_1",
            name="roll_dice",
            arguments={"notation": "1d20", "reason": "initiative"},
        )
        result = execute_tool_call(tool_call)
        assert result.success
        assert result.tool_call_id == "test_1"
        assert "rolls" in result.content

    def test_execute_tool_call_speak(self):
        tool_call = make_tool_call(
            id="test_2",
            name="speak",
            arguments={"character": "NPC", "message": "Hello there!"},
        )
        result = execute_tool_call(tool_call)
        assert result.success
        assert result.content["character"] == "NPC"

    def test_execute_tool_call_unknown(self):
        tool_call = make_tool_call(
            id="test_3",
            name="unknown_tool",
            arguments={},
        )
        result = execute_tool_call(tool_call)
        assert not result.success
        assert "Unknown tool" in result.error

    def test_execute_multiple_tool_calls(self):
        tool_calls = [
            make_tool_call(id="1", name="roll_dice", arguments={"notation": "1d6", "reason": "damage"}),
            make_tool_call(id="2", name="speak", arguments={"character": "Guard", "message": "Halt!"}),
        ]
        results = execute_tool_calls(tool_calls)
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].tool_call_id == "1"
        assert results[1].tool_call_id == "2"


class TestConversationFormatting:
    """Test conversation formatting functions."""

    def test_format_tool_result_success(self):
        result = ToolResult(
            tool_call_id="call_123",
            content={"rolls": [15], "total": 15},
            success=True,
        )
        formatted = format_tool_result_for_conversation(result)
        assert formatted["role"] == "tool"
        assert formatted["tool_call_id"] == "call_123"
        assert json.loads(formatted["content"]) == {"rolls": [15], "total": 15}

    def test_format_tool_result_error(self):
        result = ToolResult(
            tool_call_id="call_456",
            content={},
            success=False,
            error="Invalid dice notation",
        )
        formatted = format_tool_result_for_conversation(result)
        assert formatted["role"] == "tool"
        assert formatted["content"] == "Invalid dice notation"

    def test_format_tool_calls(self):
        tool_calls = [
            make_tool_call(id="call_1", name="roll_dice", arguments={"notation": "1d20", "reason": "test"}),
            make_tool_call(id="call_2", name="speak", arguments={"character": "NPC", "message": "Hi"}),
        ]
        formatted = format_tool_calls_for_conversation(tool_calls)
        assert len(formatted) == 2
        assert formatted[0]["id"] == "call_1"
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "roll_dice"
        assert json.loads(formatted[0]["function"]["arguments"])["notation"] == "1d20"
        assert formatted[1]["id"] == "call_2"
        assert formatted[1]["function"]["name"] == "speak"


class TestRPGEpisode:
    """Test the RPGEpisode dataclass."""

    def test_default_initialization(self):
        episode = RPGEpisode()
        assert episode.step_count == 0
        assert episode.game_setting is None
        assert episode.player_character is None
        assert episode.parameters_generated is False
        assert episode.scenario_generated is False
        assert episode.conversation == []
        assert episode.tool_calls_history == []

    def test_with_values(self):
        episode = RPGEpisode(
            game_setting="A dark forest",
            player_character="Elven ranger",
            step_count=3,
        )
        assert episode.game_setting == "A dark forest"
        assert episode.player_character == "Elven ranger"
        assert episode.step_count == 3


# ============================================================================
# Integration Tests - Test full flow without actual inference
# ============================================================================


class TestGenerateWithToolsLoop:
    """Test the generate_with_tools_loop function with mocked wrappers."""

    def test_single_generation_no_tools(self):
        """Test generation that returns content without tool calls."""
        async def _test():
            mock_wrapper = MagicMock()
            mock_wrapper.generate_with_tools = AsyncMock(
                return_value=[
                    GenerationResult(
                        content="The adventurer stands at the crossroads.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )
                ]
            )

            conversation = [{"role": "user", "content": "Begin the adventure."}]
            result, tool_results = await generate_with_tools_loop(
                mock_wrapper, conversation, DM_TOOLS
            )

            assert result.content == "The adventurer stands at the crossroads."
            assert result.tool_calls == []
            assert tool_results == []
            mock_wrapper.generate_with_tools.assert_called_once()

        asyncio.run(_test())

    def test_generation_with_tool_calls(self):
        """Test generation that uses tools and then completes."""
        async def _test():
            mock_wrapper = MagicMock()

            # First call returns tool calls
            first_response = GenerationResult(
                content=None,
                tool_calls=[
                    make_tool_call(
                        id="call_123",
                        name="roll_dice",
                        arguments={"notation": "1d20", "reason": "perception check"},
                    )
                ],
                stop_reason="tool_use",
            )

            # Second call returns final content
            second_response = GenerationResult(
                content="You rolled a 15 on your perception check and notice a hidden door.",
                tool_calls=[],
                stop_reason="end_turn",
            )

            mock_wrapper.generate_with_tools = AsyncMock(
                side_effect=[[first_response], [second_response]]
            )

            conversation = [{"role": "user", "content": "I search the room."}]
            result, tool_results = await generate_with_tools_loop(
                mock_wrapper, conversation, DM_TOOLS
            )

            assert result.content == "You rolled a 15 on your perception check and notice a hidden door."
            assert len(tool_results) == 1
            assert tool_results[0].tool_call_id == "call_123"
            assert "rolls" in tool_results[0].content
            assert mock_wrapper.generate_with_tools.call_count == 2

        asyncio.run(_test())

    def test_generation_with_multiple_tool_calls(self):
        """Test generation with multiple sequential tool calls."""
        async def _test():
            mock_wrapper = MagicMock()

            responses = [
                # First: roll dice
                [GenerationResult(
                    content=None,
                    tool_calls=[
                        make_tool_call(id="call_1", name="roll_dice", arguments={"notation": "1d20", "reason": "attack"})
                    ],
                    stop_reason="tool_use",
                )],
                # Second: speak
                [GenerationResult(
                    content=None,
                    tool_calls=[
                        make_tool_call(id="call_2", name="speak", arguments={"character": "Guard", "message": "Halt!"})
                    ],
                    stop_reason="tool_use",
                )],
                # Third: final response
                [GenerationResult(
                    content="The guard blocks your path.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )],
            ]

            mock_wrapper.generate_with_tools = AsyncMock(side_effect=responses)

            conversation = [{"role": "user", "content": "I approach the gate."}]
            result, tool_results = await generate_with_tools_loop(
                mock_wrapper, conversation, DM_TOOLS
            )

            assert result.content == "The guard blocks your path."
            assert len(tool_results) == 2
            assert tool_results[0].tool_call_id == "call_1"
            assert tool_results[1].tool_call_id == "call_2"

        asyncio.run(_test())

    def test_max_iterations_limit(self):
        """Test that the loop stops after max iterations."""
        async def _test():
            mock_wrapper = MagicMock()

            # Always return tool calls (never stops)
            mock_wrapper.generate_with_tools = AsyncMock(
                return_value=[
                    GenerationResult(
                        content=None,
                        tool_calls=[
                            make_tool_call(id="call_inf", name="roll_dice", arguments={"notation": "1d6", "reason": "test"})
                        ],
                        stop_reason="tool_use",
                    )
                ]
            )

            conversation = [{"role": "user", "content": "Test"}]
            result, tool_results = await generate_with_tools_loop(
                mock_wrapper, conversation, DM_TOOLS, max_iterations=3
            )

            # Should stop after 3 iterations
            assert mock_wrapper.generate_with_tools.call_count == 3
            assert len(tool_results) == 3

        asyncio.run(_test())


class TestRoleplayingTaskIntegration:
    """Integration tests for the full RoleplayingGameMultiStepTask flow."""

    def _create_mock_task(self):
        """Create a task with mocked generation wrappers."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            task = RoleplayingGameMultiStepTask(
                run_mode="cli",
                max_user_responses=2,
                num_episodes=1,
            )
            return task

    def test_start_episode(self):
        """Test episode initialization."""
        async def _test():
            mock_task = self._create_mock_task()
            episode = await mock_task.start_episode(None)

            assert isinstance(episode, RPGEpisode)
            assert episode.step_count == 0
            assert episode.parameters_generated is False
            assert episode.scenario_generated is False
            assert episode.conversation == []

        asyncio.run(_test())

    def test_generate_parameters(self):
        """Test parameter generation step."""
        async def _test():
            mock_task = self._create_mock_task()
            episode = RPGEpisode()

            # Mock the generate method for parameter generation
            mock_task.generation_wrappers["parameter"].generate = AsyncMock(
                return_value=[
                    """Here are the game parameters:

<game_setting>
A dark and mysterious forest filled with ancient magic. The trees whisper secrets
of a forgotten civilization, and strange creatures lurk in the shadows.
</game_setting>

<player_character>
Elara, a young elven ranger with emerald eyes and silver hair. She carries a bow
crafted from the heartwood of the World Tree and seeks to uncover the truth about
her missing parents.
</player_character>
"""
                ]
            )

            await mock_task._generate_parameters(episode)

            assert episode.game_setting is not None
            assert "forest" in episode.game_setting.lower()
            assert episode.player_character is not None
            assert "elara" in episode.player_character.lower() or "ranger" in episode.player_character.lower()

        asyncio.run(_test())

    def test_generate_scenario(self):
        """Test scenario generation with tool calls."""
        async def _test():
            mock_task = self._create_mock_task()
            episode = RPGEpisode(
                game_setting="A dark forest with ancient secrets.",
                player_character="Elara, an elven ranger.",
            )

            # Mock generate_with_tools to return a scenario with tool usage
            mock_task.generation_wrappers["generation"].generate_with_tools = AsyncMock(
                side_effect=[
                    # First call: DM uses speak tool
                    [GenerationResult(
                        content=None,
                        tool_calls=[
                            make_tool_call(
                                id="call_npc",
                                name="speak",
                                arguments={
                                    "character": "Old Hermit",
                                    "message": "Beware the shadows, young one...",
                                    "tone": "whispered",
                                },
                            )
                        ],
                        stop_reason="tool_use",
                    )],
                    # Second call: Final narration
                    [GenerationResult(
                        content="You stand at the edge of the ancient forest. An old hermit has just warned you of the dangers ahead.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )],
                ]
            )

            await mock_task._generate_scenario(episode, mock_task.generation_wrappers["generation"])

            assert len(episode.conversation) == 1
            assert episode.conversation[0]["role"] == "assistant"
            assert "forest" in episode.conversation[0]["content"].lower()
            assert len(episode.tool_calls_history) == 1
            assert episode.tool_calls_history[0]["turn"] == 0

        asyncio.run(_test())

    def test_generate_turn(self):
        """Test a full turn (player action + DM response)."""
        async def _test():
            mock_task = self._create_mock_task()
            episode = RPGEpisode(
                game_setting="A dark forest.",
                player_character="Elara the ranger.",
                scenario_generated=True,
                conversation=[
                    {"role": "assistant", "content": "You stand at the forest entrance."}
                ],
            )

            # Mock player action generation - use side_effect so loop terminates after tool execution
            mock_task.generation_wrappers["followup"].generate_with_tools = AsyncMock(
                side_effect=[
                    # First call: player uses action tool
                    [GenerationResult(
                        content=None,
                        tool_calls=[
                            make_tool_call(
                                id="call_action",
                                name="action",
                                arguments={"description": "draw bow and enter forest cautiously"},
                            )
                        ],
                        stop_reason="tool_use",
                    )],
                    # Second call: player finishes with content
                    [GenerationResult(
                        content="I draw my bow and cautiously enter the forest, keeping my eyes peeled for danger.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )],
                ]
            )

            # Mock DM response generation
            mock_task.generation_wrappers["generation"].generate_with_tools = AsyncMock(
                side_effect=[
                    # DM rolls perception
                    [GenerationResult(
                        content=None,
                        tool_calls=[
                            make_tool_call(
                                id="call_roll",
                                name="roll_dice",
                                arguments={"notation": "1d20+3", "reason": "perception check"},
                            )
                        ],
                        stop_reason="tool_use",
                    )],
                    # DM presents choices
                    [GenerationResult(
                        content=None,
                        tool_calls=[
                            make_tool_call(
                                id="call_choices",
                                name="present_choices",
                                arguments={
                                    "prompt": "You notice two paths ahead. Which do you take?",
                                    "choices": [
                                        {"id": "left", "description": "The overgrown path to the left"},
                                        {"id": "right", "description": "The well-worn trail to the right"},
                                    ],
                                },
                            )
                        ],
                        stop_reason="tool_use",
                    )],
                    # Final response
                    [GenerationResult(
                        content="Your keen ranger senses detect movement in the underbrush. Two paths diverge before you.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )],
                ]
            )

            success = await mock_task._generate_turn(
                episode, mock_task.generation_wrappers["generation"]
            )

            assert success is True
            assert len(episode.conversation) == 3  # Original + user + assistant
            assert episode.conversation[1]["role"] == "user"
            assert episode.conversation[2]["role"] == "assistant"

            # Check tool calls were recorded
            player_tools = [t for t in episode.tool_calls_history if t.get("role") == "player"]
            dm_tools = [t for t in episode.tool_calls_history if t.get("role") == "dm"]
            assert len(player_tools) == 1
            assert len(dm_tools) == 2  # roll_dice + present_choices

        asyncio.run(_test())

    def test_full_episode_flow(self):
        """Test the complete episode flow from start to finish."""
        async def _test():
            mock_task = self._create_mock_task()

            # Mock parameter generation
            mock_task.generation_wrappers["parameter"].generate = AsyncMock(
                return_value=[
                    "<game_setting>A haunted castle.</game_setting>\n<player_character>A brave knight.</player_character>"
                ]
            )

            # Mock scenario generation
            mock_task.generation_wrappers["generation"].generate_with_tools = AsyncMock(
                return_value=[
                    GenerationResult(
                        content="You stand before the castle gates.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )
                ]
            )

            # Mock turn generation (player + DM)
            mock_task.generation_wrappers["followup"].generate_with_tools = AsyncMock(
                return_value=[
                    GenerationResult(
                        content="I approach the gates.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )
                ]
            )

            # Start episode
            episode = await mock_task.start_episode(None)
            assert episode.parameters_generated is False

            # Step 1: Generate parameters
            result = await mock_task.step_episode(episode)
            assert result is None  # Not finished
            assert episode.parameters_generated is True
            assert episode.game_setting is not None

            # Step 2: Generate scenario
            result = await mock_task.step_episode(episode)
            assert result is None  # Not finished
            assert episode.scenario_generated is True
            assert len(episode.conversation) == 1

            # Step 3: First turn
            result = await mock_task.step_episode(episode)
            assert result is None  # Not finished (max_user_responses=2)
            assert episode.step_count == 1

            # Step 4: Second turn
            result = await mock_task.step_episode(episode)
            assert result is None
            assert episode.step_count == 2

            # Step 5: Episode complete
            result = await mock_task.step_episode(episode)
            assert result is episode  # Returns completed episode
            assert episode.step_count == 2

        asyncio.run(_test())

    def test_get_output_row(self):
        """Test output row formatting."""
        mock_task = self._create_mock_task()
        episode = RPGEpisode(
            game_setting="A magical realm.",
            player_character="A wizard named Merlin.",
            step_count=3,
            conversation=[
                {"role": "assistant", "content": "Welcome, wizard."},
                {"role": "user", "content": "I cast a spell."},
                {"role": "assistant", "content": "The spell illuminates the room."},
            ],
            tool_calls_history=[
                {"turn": 0, "tool_call_id": "call_1", "result": {"total": 15}, "success": True},
                {"turn": 1, "role": "player", "tool_call_id": "call_2", "result": {"executed": True}, "success": True},
            ],
        )

        output = mock_task.get_output_row(episode)

        assert len(output) == 1
        row = output[0]
        assert row["game_setting"] == "A magical realm."
        assert row["player_character"] == "A wizard named Merlin."
        assert row["num_turns"] == 3
        assert len(row["conversation"]) == 3
        assert len(row["tool_calls_history"]) == 2


class TestToolExecutionInContext:
    """Test tool execution with realistic game scenarios."""

    def test_dice_roll_combat_scenario(self):
        """Test dice rolling in a combat context."""
        # Attack roll
        attack_result = execute_roll_dice("1d20+5", "attack roll against goblin")
        assert attack_result.success
        assert 6 <= attack_result.content["total"] <= 25  # 1+5 to 20+5

        # Damage roll
        damage_result = execute_roll_dice("2d6+3", "longsword damage")
        assert damage_result.success
        assert 5 <= damage_result.content["total"] <= 15  # 2+3 to 12+3

    def test_random_choice_encounter(self):
        """Test random choice for encounter generation."""
        enemies = ["goblin", "orc", "skeleton", "wolf"]
        result = execute_random_choice(enemies, "random encounter")
        assert result.success
        assert result.content["chosen"] in enemies

    def test_present_choices_dialogue(self):
        """Test presenting dialogue choices."""
        choices = [
            {"id": "fight", "description": "Draw your sword and attack"},
            {"id": "talk", "description": "Attempt to negotiate"},
            {"id": "flee", "description": "Turn and run away"},
        ]
        result = execute_present_choices("The bandit blocks your path. What do you do?", choices)
        assert result.success
        assert len(result.content["choices"]) == 3
        assert result.content["awaiting_player_choice"] is True

    def test_speak_npc_dialogue(self):
        """Test NPC dialogue generation."""
        result = execute_speak(
            character="Innkeeper Greta",
            message="Welcome to the Prancing Pony! What'll it be, traveler?",
            tone="cheerfully",
        )
        assert result.success
        assert result.content["character"] == "Innkeeper Greta"
        assert "Welcome" in result.content["message"]
        assert result.content["tone"] == "cheerfully"

    def test_action_exploration(self):
        """Test player exploration action."""
        result = execute_action(
            description="search the ancient bookshelf for hidden compartments",
            target="bookshelf",
        )
        assert result.success
        assert result.content["target"] == "bookshelf"
        assert result.content["executed"] is True
