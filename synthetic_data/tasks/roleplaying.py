import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Dataset
from jinja2 import Template
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
)
from synthetic_data.generation_utils import GenerationArgs, GenerationResult
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    DUNGEON_MASTER_ACTION_PROMPT,
    GAME_PARAMETER_PROMPT,
    PLAYER_ACTION_PROMPT,
)
from synthetic_data.tasks.roleplaying_tools import (
    DM_TOOLS,
    PRESENT_CHOICES_TOOL,
    SPEAK_TOOL,
    ToolResult,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    Message,
    log_conversation,
    parse_xml_tags,
)

MAX_PARSE_RETRIES = 3
MAX_TOOL_ITERATIONS = 5

GAME_SETTINGS = [
    "forest",
    "cave",
    "dungeon",
    "city",
    "ruins",
    "castle",
    "temple",
    "library",
    "museum",
    "library",
    "museum",
]

RolePlayingRole = Literal["player", "dungeon_master"]


@dataclass
class Action:
    role: Literal["player", "dungeon_master"]
    message: Message


@dataclass
class RPGEpisode:
    step_count: int = 0
    game_setting: str | None = None
    player_character: str | None = None
    scenario_message: GenerationResult | None = None
    actions: list[Action] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


USE_DEV_MODELS = True


def format_tool_result_for_conversation(result: ToolResult) -> dict[str, Any]:
    """Format a tool result for inclusion in the conversation."""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": json.dumps(result.content) if result.success else result.error,
    }


class RoleplayingGameMultiStepTask(BaseTask[None, RPGEpisode]):
    """
    Multi-step roleplaying game task using native tool calling:
    Step 1: Generate game parameters (setting and characters)
    Step 2: Generate the initial scenario using DM tools
    Step 3: Generate turn-by-turn conversation with tool usage
    """

    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_name = "roleplaying_game_multi_step_dev"
    output_dataset_org = "roborovski"
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(
        self,
        run_mode: RunMode,
        max_user_responses: int = 10,
        num_episodes: int = 1000,
    ):
        super().__init__(run_mode)
        self.max_user_responses = max_user_responses
        self.num_episodes = num_episodes

        gen_model: RemoteModel = (
            "claude-4-5-haiku" if USE_DEV_MODELS else "claude-4-5-sonnet"
        )

        self._add_generation_wrapper("dungeon_master", gen_model)
        self._add_generation_wrapper("player", gen_model)
        self._add_generation_wrapper("adventure_parameters", gen_model)

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic for roleplaying game.
        Since this is a synthetic task, we create dummy data.
        """
        logger.info("Loading custom dataset for roleplaying game")
        seed = random.randint(0, 2**32)
        rng = random.Random(seed)
        game_setting = rng.choice(GAME_SETTINGS)
        episode_data = [
            {
                "game_setting": game_setting,
                "seed": random.randint(0, 2**32),
            }
            for _ in range(self.num_episodes)
        ]
        logger.info(f"Created {len(episode_data)} episodes")
        return Dataset.from_list(episode_data)

    async def initial_step(self, sample: None) -> RPGEpisode:
        seed = hash(str(sample)) % (2**32)
        ep = RPGEpisode()
        ep.metadata = {
            "dungeon_master_model": self.generation_wrappers[
                "dungeon_master"
            ].provider_name,
            "player_model": self.generation_wrappers["player"].provider_name,
            "adventure_parameters_model": self.generation_wrappers[
                "adventure_parameters"
            ].provider_name,
            "max_user_responses": self.max_user_responses,
            "seed": seed,
        }

        # Generate parameters and scenario in parallel
        (game_setting, player_character), scenario_result = await asyncio.gather(
            self._generate_parameters(ep),
            self._generate_scenario(ep, self.generation_wrappers["dungeon_master"]),
        )
        ep.game_setting = game_setting
        ep.player_character = player_character
        ep.scenario_message = scenario_result
        return ep

    async def step(self, episode: RPGEpisode) -> tuple[RPGEpisode, bool]:
        # Generate player action
        conversation = self._format_conversation(episode, "player")
        log_conversation(conversation)
        results = await self.generation_wrappers["player"].generate([conversation])
        assert len(results) == 1
        episode.actions.append(Action(role="player", message=results[0].message))

        # Generate dungeon master action
        conversation = self._format_conversation(episode, "dungeon_master")
        results = await self.generation_wrappers["dungeon_master"].generate(
            [conversation],
            args=GenerationArgs(max_tokens=1024, tools=DM_TOOLS),
        )
        assert len(results) == 1
        episode.actions.append(
            Action(role="dungeon_master", message=results[0].message)
        )

        episode.step_count += 1
        finished = episode.step_count >= self.max_user_responses
        return episode, finished

    def format_episode(self, episode: RPGEpisode) -> dict:
        """Convert a finished RPGEpisode to a dictionary for storage."""

        def serialize_tool_calls(tool_calls: list | None) -> list | None:
            if not tool_calls:
                return None
            return [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]

        def serialize_message(msg: Message) -> dict:
            result = {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
            }
            if "tool_calls" in msg:
                result["tool_calls"] = serialize_tool_calls(msg["tool_calls"])
            return result

        # Serialize scenario_message
        scenario_dict = None
        if episode.scenario_message:
            scenario_dict = {
                "content": episode.scenario_message.content,
                "tool_calls": serialize_tool_calls(episode.scenario_message.tool_calls),
                "finish_reason": episode.scenario_message.finish_reason,
            }

        # Serialize actions
        actions_list = [
            {
                "role": action.role,
                "message": serialize_message(action.message),
            }
            for action in episode.actions
        ]

        return {
            "step_count": episode.step_count,
            "game_setting": episode.game_setting,
            "player_character": episode.player_character,
            "scenario_message": scenario_dict,
            "actions": actions_list,
            "metadata": episode.metadata,
        }

    async def _generate_parameters(
        self, episode: RPGEpisode
    ) -> tuple[str, str] | tuple[None, None]:
        """Generate game parameters (setting and characters) using XML parsing."""
        theme_input = episode.scenario_message or "A mysterious adventure"
        parameter_conversation: Conversation = [
            {
                "role": "system",
                "content": GAME_PARAMETER_PROMPT,
            },
            {
                "role": "user",
                "content": f"Generate game parameters for a roleplaying scenario based on this theme: {theme_input}",
            },
        ]

        logger.info(
            f"Generating game parameters with {self.generation_model_names['adventure_parameters']}"
        )

        for attempt in range(MAX_PARSE_RETRIES):
            results = await self.generation_wrappers["adventure_parameters"].generate(
                [parameter_conversation]
            )
            try:
                content = results[0].content or ""
                parsed_tags = parse_xml_tags(
                    content, required_tags=["game_setting", "player_character"]
                )
                game_setting = parsed_tags["game_setting"]
                player_character = parsed_tags["player_character"]
                logger.debug(f"\nGame setting:\n{game_setting}")
                logger.debug(f"\nPlayer character:\n{player_character}")
                return game_setting, player_character
            except ValueError as e:
                if attempt < MAX_PARSE_RETRIES - 1:
                    logger.warning(
                        f"Parse failed (attempt {attempt + 1}), retrying: {e}"
                    )
                else:
                    raise ValueError(
                        f"Failed to generate game parameters after {MAX_PARSE_RETRIES} attempts"
                    )
        return None, None

    def _game_master_system_prompt(self, episode: RPGEpisode) -> str:
        template: Template = Template(DUNGEON_MASTER_ACTION_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTER=episode.player_character or "A brave adventurer",
        )
        return formatted_prompt

    def _player_system_prompt(self, episode: RPGEpisode) -> str:
        template: Template = Template(PLAYER_ACTION_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTER=episode.player_character or "A brave adventurer",
        )
        return formatted_prompt

    def _format_tool_calls_for_user_message(
        self, tool_calls: list[Any] | None, existing_content: str = ""
    ) -> str:
        if not tool_calls:
            return existing_content

        serialized_tool_calls = [
            tc
            if isinstance(tc, dict)
            else {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        return (
            existing_content
            + f"\n<tool_calls>{json.dumps(serialized_tool_calls)}</tool_calls>"
        )

    async def _generate_scenario(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ) -> GenerationResult:
        """Generate the initial roleplaying scenario using DM tools."""
        scenario_conversation: Conversation = [
            {
                "role": "system",
                "content": self._game_master_system_prompt(episode),
            },
            {
                "role": "user",
                "content": "Begin the adventure. Set the scene and introduce the player to their situation.",
            },
        ]

        results = await generation_wrapper.generate(
            [scenario_conversation],
            args=GenerationArgs(
                max_tokens=1024, tools=[SPEAK_TOOL, PRESENT_CHOICES_TOOL]
            ),
        )
        return results[0]

    def _format_conversation(
        self, episode: RPGEpisode, generating_role: RolePlayingRole
    ) -> Conversation:
        """
        For the DM, generate with player actions as user actions, and DM actions as assistant actions.
        For the player, generate with DM actions as user actions, and player actions as assistant actions.
        """
        system_prompt = (
            self._game_master_system_prompt(episode)
            if generating_role == "dungeon_master"
            else self._player_system_prompt(episode)
        )
        conversation: Conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        assert episode.scenario_message is not None

        # Add scenario to conversation with proper role based on perspective
        scenario_msg = episode.scenario_message.message
        if generating_role == "player":
            # From player perspective, scenario (from DM) is a user message
            # Tool calls need to be formatted as content since user messages can't have tool_calls
            scenario_content = self._format_tool_calls_for_user_message(
                scenario_msg.get("tool_calls"), scenario_msg.get("content", "")
            )
            conversation.append({"role": "user", "content": scenario_content})
        else:
            # From DM perspective, scenario stays as assistant message
            conversation.append(scenario_msg)

        for action in episode.actions:
            if generating_role == "dungeon_master":
                # DM perspective: player -> user, dm -> assistant
                role = "user" if action.role == "player" else "assistant"
            else:  # generating_role == "player"
                # Player perspective: dungeon_master -> user, player -> assistant
                role = "user" if action.role == "dungeon_master" else "assistant"

            message: Message = {
                "role": role,
                "content": action.message.get("content", ""),
            }

            tool_calls = action.message.get("tool_calls")
            # Only assistant messages can have tool_calls in the OpenAI API format
            if tool_calls and role == "assistant":
                # If assistant, add tool calls
                message["tool_calls"] = tool_calls
            elif tool_calls:
                # If user, format tool call in message content
                # TODO determine better way to format this to not break tool call formatting
                # in the fine-tuned model
                message["content"] = self._format_tool_calls_for_user_message(
                    tool_calls, message.get("content", "")
                )

            conversation.append(message)

        return conversation
