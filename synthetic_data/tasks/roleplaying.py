import json
import random
from typing import Literal, NotRequired, TypedDict

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
    format_tool_result_as_nl,
    format_tool_use_as_nl,
)
from synthetic_data.utils import (
    ContentBlock,
    Conversation,
    DatasetFormat,
    Message,
    TextBlock,
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

RolePlayingRole = Literal["player", "dungeon_master", "tool_result"]


class Action(TypedDict):
    role: RolePlayingRole
    message: Message
    tool_calling_role: NotRequired[RolePlayingRole | None]


class RPGEpisode(TypedDict):
    game_setting: str | None
    player_character: str | None
    actions: list[Action]
    metadata: dict


def create_episode() -> RPGEpisode:
    """Create a new empty RPGEpisode."""
    return RPGEpisode(
        game_setting=None,
        player_character=None,
        actions=[],
        metadata={},
    )


def create_action(
    role: RolePlayingRole,
    message: Message,
    tool_calling_role: RolePlayingRole | None = None,
) -> Action:
    """Create a new Action."""
    action: Action = {"role": role, "message": message}
    if tool_calling_role is not None:
        action["tool_calling_role"] = tool_calling_role
    return action


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
        max_steps: int = 20,
        num_episodes: int = 1000,
    ):
        super().__init__(run_mode)
        self.max_steps = max_steps
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
        ep = create_episode()
        ep["metadata"] = {
            "dungeon_master_model": self.generation_wrappers[
                "dungeon_master"
            ].provider_name,
            "player_model": self.generation_wrappers["player"].provider_name,
            "adventure_parameters_model": self.generation_wrappers[
                "adventure_parameters"
            ].provider_name,
            "max_user_responses": self.max_steps,
            "seed": seed,
        }
        game_setting, player_character = await self._generate_setting_and_characters(ep)

        assert game_setting is not None and player_character is not None
        ep["game_setting"] = game_setting
        ep["player_character"] = player_character
        return ep

    def _get_new_actions(
        self,
        conversation: Conversation,
        initial_length: int,
        role: RolePlayingRole,
    ) -> list[Action]:
        """
        Extract all new messages added during generation, preserving tool_use blocks.

        Returns a list of Actions for:
        - Assistant messages (may contain tool_use blocks like dice rolls)
        - User messages containing tool_result blocks
        """
        actions: list[Action] = []
        new_messages = conversation[initial_length:]

        for msg in new_messages:
            msg_role = msg.get("role")
            content = msg.get("content", [])

            if msg_role == "assistant":
                # Save all assistant messages (includes tool_use and text blocks)
                actions.append(create_action(role=role, message=msg))
            elif msg_role == "user":
                # Check if this is a tool_result message (from tool execution)
                has_tool_result = any(
                    block.get("type") == "tool_result" for block in content
                )
                if has_tool_result:
                    # Save tool results as part of the same role's turn
                    # Store which role performed the tool call that produced this result
                    actions.append(
                        create_action(
                            role="tool_result", message=msg, tool_calling_role=role
                        )
                    )

        return actions

    async def step(self, episode: RPGEpisode) -> tuple[RPGEpisode, bool]:
        # On the first turn, DM goes first to set the scene
        # After that, player goes first, then DM responds
        if len(episode["actions"]) == 0:
            # Generate dungeon master action to set the scene
            conversation = self._format_conversation(episode, "dungeon_master")
            initial_length = len(conversation)
            result = await self.generation_wrappers["dungeon_master"].generate(
                conversation,
                args=GenerationArgs(
                    max_tokens=1024,
                    tools=DM_TOOLS,
                    tool_use_executor=execute_tool_use_block,
                    tool_choice={
                        "type": "any"
                    },  # Force DM to use tools (roll_dice, speak, etc.)
                ),
            )
            # Extract all new messages (tool_use, tool_result, and final text)
            new_actions = self._get_new_actions(
                result.conversation, initial_length, "dungeon_master"
            )
            episode["actions"].extend(new_actions)

            return episode, False

        # Generate player action
        conversation = self._format_conversation(episode, "player")
        initial_length = len(conversation)

        result = await self.generation_wrappers["player"].generate(
            conversation,
            args=GenerationArgs(
                max_tokens=1024,
                tools=PLAYER_TOOLS,
                tool_use_executor=execute_tool_use_block,
            ),
        )
        # Extract all new messages from player generation
        new_actions = self._get_new_actions(
            result.conversation, initial_length, "player"
        )
        episode["actions"].extend(new_actions)
        logger.info("Player conversation:")
        log_conversation(self._format_conversation(episode, "player"))

        # Generate dungeon master action
        conversation = self._format_conversation(episode, "dungeon_master")
        initial_length = len(conversation)
        result = await self.generation_wrappers["dungeon_master"].generate(
            conversation,
            args=GenerationArgs(
                max_tokens=1024,
                tools=DM_TOOLS,
                tool_use_executor=execute_tool_use_block,
                tool_choice={"type": "any"},  # Force DM to use tools
            ),
        )
        # Extract all new messages from DM generation
        new_actions = self._get_new_actions(
            result.conversation, initial_length, "dungeon_master"
        )
        episode["actions"].extend(new_actions)
        logger.info("Dungeon master conversation:")
        log_conversation(self._format_conversation(episode, "dungeon_master"))

        finished = len(episode["actions"]) >= self.max_steps
        return episode, finished

    def format_episode(self, episode: RPGEpisode) -> dict:
        """Convert a finished RPGEpisode to a dictionary for storage.

        Note: We serialize the message content to JSON strings to avoid schema
        mismatches in parquet - different tools have different input schemas
        (e.g., action has 'description', roll_dice has 'notation' and 'reason').
        """
        serialized_actions = []

        for action in episode["actions"]:
            # Serialize content blocks to JSON string to avoid schema issues
            message_copy = {
                "role": action["message"].get("role"),
                "content": json.dumps(action["message"].get("content", [])),
            }
            action_dict = {"role": action["role"], "message": message_copy}
            if action.get("tool_calling_role") is not None:
                action_dict["tool_calling_role"] = action.get("tool_calling_role")
            serialized_actions.append(action_dict)

        return {
            "game_setting": episode["game_setting"],
            "player_character": episode["player_character"],
            "actions": serialized_actions,
            "metadata": episode["metadata"],
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
        """
        For tool calls that are now user turns, format them as natural language.
        """
        out_blocks: list[ContentBlock] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "tool_use":
                # Format tool_use as natural language for user turns
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                nl_text = format_tool_use_as_nl(tool_name, tool_input)
                text_block: TextBlock = {
                    "type": "text",
                    "text": nl_text,
                }
                out_blocks.append(text_block)
            elif block_type == "tool_result":
                # Convert tool_result to NL text for cross-perspective viewing
                tool_name = block.get("name", "")
                content = block.get("content", "")
                # Try to parse JSON and format as NL
                try:
                    if isinstance(content, str) and content.strip().startswith("{"):
                        result_obj = json.loads(content)
                        nl_text = format_tool_result_as_nl(tool_name, result_obj)
                    else:
                        # Already NL or plain text
                        nl_text = content
                except (json.JSONDecodeError, TypeError):
                    nl_text = content
                text_block: TextBlock = {
                    "type": "text",
                    "text": nl_text,
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

        assert episode["game_setting"] is not None
        assert episode["player_character"] is not None

        player_system_prompt = player_action_prompt(
            episode["game_setting"], episode["player_character"]
        )
        dm_system_prompt = dm_action_prompt(
            episode["game_setting"], episode["player_character"]
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
        if generating_role == "dungeon_master" and not episode["actions"]:
            content_block: TextBlock = {
                "type": "text",
                "text": "Begin the adventure. Set the opening scene.",
            }
            conversation.append({"role": "user", "content": [content_block]})

        for action in episode["actions"]:
            # Skip tool_result actions when formatting conversation
            # Tool results are intermediate messages that were processed during generation
            # They're saved in the episode for the output dataset, but shouldn't be
            # included when constructing the conversation for the next API call
            if action["role"] == "tool_result":
                continue

            if generating_role == "dungeon_master":
                # DM perspective: player -> user, dm -> assistant
                role = "user" if action["role"] == "player" else "assistant"
            else:  # generating_role == "player"
                # Player perspective: dungeon_master -> user, player -> assistant
                role = "user" if action["role"] == "dungeon_master" else "assistant"

            content = action["message"].get("content", [])

            # Convert tool_use/tool_result blocks to text for ALL messages
            # This avoids API errors about tool_use requiring matching tool_result
            # The raw tool calls are still preserved in the episode["actions"] for the output dataset
            content = self._convert_tool_calls_to_text_blocks(content)

            # Skip messages with empty content
            if not content:
                continue

            message: Message = {
                "role": role,
                "content": content,
            }

            conversation.append(message)

        return conversation
