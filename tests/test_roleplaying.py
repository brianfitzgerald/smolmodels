"""Comprehensive integration tests for the RoleplayingGameMultiStepTask."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synthetic_data.generation_utils import GenerationResult
from synthetic_data.tasks.roleplaying import (
    Action,
    RoleplayingGameMultiStepTask,
    RPGEpisode,
)
from synthetic_data.utils import Message


class TestRoleplayingGameMultiStepTask:
    """Integration tests for the full roleplaying game flow."""

    @pytest.fixture
    def task(self):
        """Create a task instance with mocked generation wrappers."""
        with patch("synthetic_data.tasks.get_generation_wrapper") as mock_get_wrapper:
            # Create mock wrappers
            mock_wrapper = MagicMock()
            mock_wrapper.provider_name = "mock"
            mock_get_wrapper.return_value = mock_wrapper
            task = RoleplayingGameMultiStepTask(run_mode="cli", num_episodes=5)
            return task

    def test_load_custom_creates_episodes(self, task):
        """Test that load_custom creates the correct number of episodes."""
        dataset = task.load_custom("./dataset_files")

        assert len(dataset) == 5
        assert "game_setting" in dataset.column_names
        assert "seed" in dataset.column_names
        # Each episode should have a game setting from the predefined list
        for row in dataset:
            assert row["game_setting"] in [
                "forest",
                "cave",
                "dungeon",
                "city",
                "ruins",
                "castle",
                "temple",
                "library",
                "museum",
            ]

    def test_format_conversation_dm_perspective(self, task):
        """Test conversation formatting from dungeon master's perspective."""
        # Create a scenario message for the episode
        scenario_result = GenerationResult(
            content="You find yourself at the entrance of a dark forest.",
            tool_calls=[],
        )

        episode = RPGEpisode(
            game_setting="A dark forest",
            player_character="A brave warrior",
            scenario_message=scenario_result,
        )
        player_message: Message = {
            "role": "user",
            "content": "I search the room.",
        }
        dm_message: Message = {
            "role": "assistant",
            "content": "You find a hidden treasure chest!",
        }
        episode.actions = [
            Action(role="player", message=player_message),
            Action(role="dungeon_master", message=dm_message),
        ]

        conversation = task._format_conversation(episode, "dungeon_master")

        # Should have: system, scenario, player action, dm action
        assert len(conversation) == 4
        assert conversation[0]["role"] == "system"
        assert conversation[1]["role"] == "assistant"  # scenario message
        assert conversation[2]["role"] == "user"  # player action
        assert conversation[2]["content"] == "I search the room."
        assert conversation[3]["role"] == "assistant"  # dm action
        assert conversation[3]["content"] == "You find a hidden treasure chest!"

    def test_format_conversation_player_perspective(self, task):
        """Test conversation formatting from player's perspective."""
        scenario_result = GenerationResult(
            content="You find yourself at the entrance of a dark forest.",
            tool_calls=[],
        )

        episode = RPGEpisode(
            game_setting="A dark forest",
            player_character="A brave warrior",
            scenario_message=scenario_result,
        )
        player_message: Message = {
            "role": "user",
            "content": "I search the room.",
        }
        dm_message: Message = {
            "role": "assistant",
            "content": "You find a hidden treasure chest!",
        }
        episode.actions = [
            Action(role="dungeon_master", message=dm_message),
            Action(role="player", message=player_message),
        ]

        conversation = task._format_conversation(episode, "player")

        # Should have: system, scenario (as user from player POV), dm action (as user), player action (as assistant)
        assert len(conversation) == 4
        assert conversation[0]["role"] == "system"
        assert conversation[1]["role"] == "user"  # scenario message (from DM, so user from player perspective)
        assert conversation[2]["role"] == "user"  # dm action from player perspective
        assert conversation[2]["content"] == "You find a hidden treasure chest!"
        assert conversation[3]["role"] == "assistant"  # player action
        assert conversation[3]["content"] == "I search the room."


class TestGenerateParameters:
    """Tests for the _generate_parameters method."""

    @pytest.fixture
    def task_with_mock_wrapper(self):
        """Create task with mocked generation wrappers for parameter generation."""
        with patch("synthetic_data.tasks.get_generation_wrapper") as mock_get_wrapper:
            mock_wrapper = MagicMock()
            mock_wrapper.provider_name = "mock"
            mock_get_wrapper.return_value = mock_wrapper
            task = RoleplayingGameMultiStepTask(run_mode="cli")
            return task

    @pytest.mark.asyncio
    async def test_generate_parameters_parses_xml_correctly(
        self, task_with_mock_wrapper
    ):
        """Test that XML tags are correctly parsed from the response."""
        task = task_with_mock_wrapper

        # Create mock response with XML tags
        mock_response = GenerationResult(
            content="""Here are the game parameters:
<game_setting>
A mystical forest where ancient trees whisper secrets to those who listen.
The air is thick with magic and danger lurks in every shadow.
</game_setting>

<player_character>
Aelindra, a young elven mage seeking to uncover the truth about her missing mentor.
She is skilled in illusion magic but struggles with combat spells.
</player_character>""",
            tool_calls=[],
        )

        mock_wrapper = AsyncMock()
        mock_wrapper.generate = AsyncMock(return_value=[mock_response])
        mock_wrapper.provider_name = "mock"
        task.generation_wrappers["adventure_parameters"] = mock_wrapper

        episode = RPGEpisode()
        game_setting, player_character = await task._generate_parameters(episode)

        assert game_setting is not None
        assert "mystical forest" in game_setting
        assert player_character is not None
        assert "Aelindra" in player_character


class TestStepEpisode:
    """Tests for the step method."""

    @pytest.fixture
    def task_with_mocks(self):
        """Create task with all generation wrappers mocked."""
        with patch("synthetic_data.tasks.get_generation_wrapper") as mock_get_wrapper:
            mock_wrapper = MagicMock()
            mock_wrapper.provider_name = "mock"
            mock_get_wrapper.return_value = mock_wrapper
            task = RoleplayingGameMultiStepTask(
                run_mode="cli", max_user_responses=3, num_episodes=1
            )
            return task

    @pytest.mark.asyncio
    async def test_step_generates_player_and_dm_actions(self, task_with_mocks):
        """Test that step generates both player and DM actions."""
        task = task_with_mocks

        player_response = GenerationResult(
            content="I draw my sword and approach cautiously.",
            tool_calls=[],
        )
        dm_response = GenerationResult(
            content="As you approach, the goblin notices you and hisses menacingly.",
            tool_calls=[],
        )

        player_wrapper = AsyncMock()
        player_wrapper.generate = AsyncMock(return_value=[player_response])
        player_wrapper.provider_name = "mock"

        dm_wrapper = AsyncMock()
        dm_wrapper.generate = AsyncMock(return_value=[dm_response])
        dm_wrapper.provider_name = "mock"

        task.generation_wrappers["player"] = player_wrapper
        task.generation_wrappers["dungeon_master"] = dm_wrapper

        # Create episode with scenario_message set
        scenario_result = GenerationResult(
            content="You enter a goblin-infested cave.",
            tool_calls=[],
        )

        episode = RPGEpisode(
            step_count=0,
            game_setting="A goblin-infested cave",
            player_character="A dwarven fighter",
            scenario_message=scenario_result,
        )

        updated_episode, finished = await task.step(episode)

        assert updated_episode is not None
        assert len(updated_episode.actions) == 2
        assert updated_episode.actions[0].role == "player"
        assert updated_episode.actions[1].role == "dungeon_master"
        assert updated_episode.step_count == 1
        assert finished is False  # max_user_responses is 3, step_count is 1

    @pytest.mark.asyncio
    async def test_step_returns_finished_when_max_reached(self, task_with_mocks):
        """Test that step returns finished=True when max_user_responses is reached."""
        task = task_with_mocks
        task.max_user_responses = 1  # Set low so we finish after one step

        mock_response = GenerationResult(content="Action", tool_calls=[])
        mock_wrapper = AsyncMock()
        mock_wrapper.generate = AsyncMock(return_value=[mock_response])
        mock_wrapper.provider_name = "mock"

        task.generation_wrappers["player"] = mock_wrapper
        task.generation_wrappers["dungeon_master"] = mock_wrapper

        scenario_result = GenerationResult(
            content="The adventure begins.",
            tool_calls=[],
        )

        episode = RPGEpisode(
            step_count=0,
            scenario_message=scenario_result,
        )

        updated_episode, finished = await task.step(episode)

        assert updated_episode.step_count == 1
        assert finished is True
