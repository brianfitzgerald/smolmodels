import json
import random
from dataclasses import dataclass, field
from typing import Literal

from datasets import Dataset
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
)
from synthetic_data.generation_utils import GenerationArgs, GenerationResult
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    dm_action_prompt,
    game_parameter_prompt,
    player_action_prompt,
)
from synthetic_data.tasks.roleplaying_tools import (
    DM_TOOLS,
    PLAYER_TOOLS,
    execute_tool_use_block,
)
from synthetic_data.utils import (
    ContentBlock,
    Conversation,
    DatasetFormat,
    Message,
    TextBlock,
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
    actions: list[Action] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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

        self._add_generation_wrapper("dungeon_master", "claude-4-5-sonnet")
        self._add_generation_wrapper("player", "claude-4-5-sonnet")
        self._add_generation_wrapper("adventure_parameters", "claude-4-5-haiku")

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
        game_setting, player_character = await self._generate_setting_and_characters(ep)

        assert game_setting is not None and player_character is not None
        ep.game_setting = game_setting
        ep.player_character = player_character
        return ep

    async def step(self, episode: RPGEpisode) -> tuple[RPGEpisode, bool]:
        # On the first turn, DM goes first to set the scene
        # After that, player goes first, then DM responds
        if episode.step_count == 0:
            # Generate dungeon master action to set the scene
            conversation = self._format_conversation(episode, "dungeon_master")
            result = await self.generation_wrappers["dungeon_master"].generate(
                conversation,
                args=GenerationArgs(
                    max_tokens=1024,
                    tools=DM_TOOLS,
                    tool_use_executor=execute_tool_use_block,
                    tool_choice={"type": "any"},  # Force DM to use tools (roll_dice, speak, etc.)
                ),
            )
            # Get the assistant message from the conversation (last message added by generate)
            dm_message = result.conversation[-1]
            episode.actions.append(Action(role="dungeon_master", message=dm_message))

            episode.step_count += 1
            finished = episode.step_count >= self.max_user_responses
            return episode, finished

        # Generate player action
        conversation = self._format_conversation(episode, "player")

        result = await self.generation_wrappers["player"].generate(
            conversation,
            args=GenerationArgs(
                max_tokens=1024,
                tools=PLAYER_TOOLS,
                tool_use_executor=execute_tool_use_block,
            ),
        )
        # Get the assistant message from the conversation (last message added by generate)
        assistant_message = result.conversation[-1]
        episode.actions.append(Action(role="player", message=assistant_message))

        # Generate dungeon master action
        conversation = self._format_conversation(episode, "dungeon_master")
        result = await self.generation_wrappers["dungeon_master"].generate(
            conversation,
            args=GenerationArgs(
                max_tokens=1024,
                tools=DM_TOOLS,
                tool_use_executor=execute_tool_use_block,
                tool_choice={"type": "any"},  # Force DM to use tools (roll_dice, speak, etc.)
            ),
        )
        # Get the assistant message from the conversation (last message added by generate)
        dm_message = result.conversation[-1]
        episode.actions.append(Action(role="dungeon_master", message=dm_message))

        episode.step_count += 1
        finished = episode.step_count >= self.max_user_responses
        return episode, finished

    def format_episode(self, episode: RPGEpisode) -> dict:
        """Convert a finished RPGEpisode to a dictionary for storage."""
        return {
            "step_count": episode.step_count,
            "game_setting": episode.game_setting,
            "player_character": episode.player_character,
            "actions": [action.message for action in episode.actions],
            "metadata": episode.metadata,
        }

    async def _generate_setting_and_characters(
        self, episode: RPGEpisode
    ) -> tuple[str, str] | tuple[None, None]:
        """Generate game parameters (setting and characters) using XML parsing."""
        theme_input = "A mysterious adventure"
        system_text_block: TextBlock = {"type": "text", "text": game_parameter_prompt()}
        user_text_block: TextBlock = {
            "type": "text",
            "text": f"Generate game parameters for a roleplaying scenario based on this theme: {theme_input}",
        }
        parameter_conversation: Conversation = [
            {
                "role": "system",
                "content": [system_text_block],
            },
            {
                "role": "user",
                "content": [user_text_block],
            },
        ]

        logger.info(
            f"Generating game parameters with {self.generation_model_names['adventure_parameters']}"
        )

        for attempt in range(MAX_PARSE_RETRIES):
            result = await self.generation_wrappers["adventure_parameters"].generate(
                parameter_conversation,
            )
            try:
                # Get the assistant message from the conversation (last message added by generate)
                assistant_message = result.conversation[-1]
                content = assistant_message.get("content", [])[0].get("text", "")
                parsed_tags = parse_xml_tags(
                    content, required_tags=["game_setting", "player_character"]
                )
                game_setting = parsed_tags["game_setting"]
                player_character = parsed_tags["player_character"]
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

    def _convert_tool_calls_to_text_blocks(
        self,
        content_blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        out_blocks: list[ContentBlock] = []
        for block in content_blocks:
            if block["type"] == "tool_use":
                block_sanitized = block.copy()
                del block_sanitized["type"]
                del block_sanitized["id"]
                tool_call_json_str = json.dumps(block_sanitized)
                text_block: TextBlock = {
                    "type": "text",
                    "text": f"<user_tool_call>{tool_call_json_str}</user_tool_call>",
                }
                out_blocks.append(text_block)
            elif block["type"] == "tool_result":
                # Convert tool_result to text block for cross-perspective viewing
                text_block: TextBlock = {
                    "type": "text",
                    "text": f"<tool_result>{block.get('content', '')}</tool_result>",
                }
                out_blocks.append(text_block)
            else:
                out_blocks.append(block)
        return out_blocks

    async def _generate_initial_scenario(
        self,
        game_setting: str,
        player_character: str,
        generation_wrapper: GenerationWrapper,
    ) -> GenerationResult:
        """Generate the initial roleplaying scenario using DM tools."""
        system_text_block: TextBlock = {
            "type": "text",
            "text": dm_action_prompt(game_setting, player_character),
        }
        user_text_block: TextBlock = {
            "type": "text",
            "text": "Begin the adventure. Set the scene and introduce the player to their situation.",
        }
        scenario_conversation: Conversation = [
            {
                "role": "system",
                "content": [system_text_block],
            },
            {
                "role": "user",
                "content": [user_text_block],
            },
        ]

        result = await generation_wrapper.generate(
            scenario_conversation,
            args=GenerationArgs(
                max_tokens=1024,
                tools=DM_TOOLS,
                tool_use_executor=execute_tool_use_block,
            ),
        )
        return result

    def _format_conversation(
        self, episode: RPGEpisode, generating_role: RolePlayingRole
    ) -> Conversation:
        """
        For the DM, generate with player actions as user actions, and DM actions as assistant actions.
        For the player, generate with DM actions as user actions, and player actions as assistant actions.
        """

        assert episode.game_setting is not None
        assert episode.player_character is not None

        player_system_prompt = player_action_prompt(
            episode.game_setting, episode.player_character
        )
        dm_system_prompt = dm_action_prompt(
            episode.game_setting, episode.player_character
        )
        system_text_block: TextBlock = {
            "type": "text",
            "text": dm_system_prompt
            if generating_role == "dungeon_master"
            else player_system_prompt,
        }
        conversation: Conversation = [
            {
                "role": "system",
                "content": [system_text_block],
            },
        ]

        # If generating for DM and no actions yet, add a user message to start the game
        if generating_role == "dungeon_master" and not episode.actions:
            content_block: TextBlock = {
                "type": "text",
                "text": "Begin the adventure. Set the opening scene.",
            }
            conversation.append({"role": "user", "content": [content_block]})

        for action in episode.actions:
            if generating_role == "dungeon_master":
                # DM perspective: player -> user, dm -> assistant
                role = "user" if action.role == "player" else "assistant"
            else:  # generating_role == "player"
                # Player perspective: dungeon_master -> user, player -> assistant
                role = "user" if action.role == "dungeon_master" else "assistant"

            content = action.message["content"]

            # Convert tool_use and tool_result blocks when role is "user"
            # since user messages can't contain tool_use blocks in the Anthropic API
            if role == "user":
                content = self._convert_tool_calls_to_text_blocks(content)
            else:
                # For assistant messages, filter out tool_result blocks
                # (they can only be in user messages per Anthropic API)
                content = [
                    block for block in content if block.get("type") != "tool_result"
                ]

            # Skip messages with empty content (except final assistant message)
            if not content:
                continue

            message: Message = {
                "role": role,
                "content": content,
            }

            conversation.append(message)

        return conversation
