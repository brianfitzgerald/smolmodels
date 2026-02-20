import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from datasets import Dataset
from loguru import logger

from synthetic_data.generation_utils import GenerationArgs, RemoteModel
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    dm_action_prompt,
    game_parameter_prompt,
    player_action_prompt,
)
from synthetic_data.tasks.roleplaying_tools import (
    DM_TOOLS,
    execute_tool_use_block,
)
from synthetic_data.utils import (
    ContentBlock,
    Conversation,
    DatasetFormat,
    Message,
    TextBlock,
    ToolParam,
)

MAX_PARSE_RETRIES = 3
MAX_SETTING_WORDS = 42
MAX_CHARACTER_WORDS = 28
TURN_MAX_TOKENS_BY_ROLE: dict[str, int] = {
    "dungeon_master": 512,
    "player": 192,
}
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
]
CHARACTER_ARCHETYPES = [
    "resourceful scout with a shortbow and rope kit",
    "battle-worn sellsword with a shield and torch",
    "quick-fingered infiltrator with lockpicks and smoke bombs",
    "field medic with a mace, bandages, and grit",
    "arcane skirmisher with a wand and emergency charms",
]

RolePlayingRole = Literal["player", "dungeon_master"]
DICE_OR_ROLL_TEXT_RE = re.compile(
    r"\b(\d*d\d+|roll(?:ed|ing)?|dice|d20|d12|d10|d8|d6|d4)\b", re.IGNORECASE
)


@dataclass
class Action:
    role: Literal["player", "dungeon_master"]
    messages: list[Message]


@dataclass
class RPGEpisode:
    step_count: int = 0
    game_setting: str | None = None
    player_character: str | None = None
    actions: list[Action] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class RoleplayingGameMultiStepTask(BaseTask[dict, RPGEpisode]):
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
        dungeon_master_model: RemoteModel = "claude-4-5-sonnet",
        player_model: RemoteModel = "claude-4-5-haiku",
        adventure_parameters_model: RemoteModel = "claude-4-5-haiku",
    ):
        super().__init__(run_mode)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dataset_name = f"roleplaying_game_multi_step_dev_{timestamp}"
        self.max_user_responses = max_user_responses
        self.num_episodes = num_episodes

        self._add_generation_wrapper("dungeon_master", dungeon_master_model)  # type: ignore[arg-type]
        self._add_generation_wrapper("player", player_model)  # type: ignore[arg-type]
        self._add_generation_wrapper(
            "adventure_parameters",
            adventure_parameters_model,  # type: ignore[arg-type]
        )
        for role, wrapper in self.generation_wrappers.items():
            wrapper.set_max_concurrent(1)
            # Keep latency bounded so one slow provider call does not stall a full run.
            wrapper.gen_wrapper_args.request_timeout_s = (
                30.0 if role == "player" else 60.0
            )

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic for roleplaying game.
        Since this is a synthetic task, we create dummy data.
        """
        logger.info("Loading custom dataset for roleplaying game")
        seed = random.randint(0, 2**32)
        rng = random.Random(seed)
        episode_data = [
            {
                "game_setting": rng.choice(GAME_SETTINGS),
                "seed": rng.randint(0, 2**32 - 1),
            }
            for _ in range(self.num_episodes)
        ]
        logger.info(f"Created {len(episode_data)} episodes")
        return Dataset.from_list(episode_data)

    async def initial_step(self, sample: dict) -> RPGEpisode:
        seed = int(sample.get("seed", random.randint(0, 2**32 - 1)))
        game_theme = str(sample.get("game_setting", "mysterious ruins")).strip()
        ep = RPGEpisode()
        ep.metadata = {
            "dungeon_master_model": self.generation_model_names["dungeon_master"],
            "player_model": self.generation_model_names["player"],
            "adventure_parameters_model": self.generation_model_names[
                "adventure_parameters"
            ],
            "max_user_responses": self.max_user_responses,
            "seed": seed,
            "seed_theme": game_theme,
            "generation_metrics": [],
        }
        game_setting, player_character = await self._generate_setting_and_characters(
            theme_input=game_theme,
        )

        assert game_setting is not None and player_character is not None
        ep.game_setting = game_setting
        ep.player_character = player_character
        return ep

    async def step(self, episode: RPGEpisode) -> tuple[RPGEpisode, bool]:
        if episode.step_count == 0:
            await self._run_turn(
                episode,
                role="dungeon_master",
                tools=DM_TOOLS,
            )
            episode.step_count += 1
            return episode, episode.step_count >= self.max_user_responses

        await self._run_turn(
            episode,
            role="player",
            tools=[],
        )
        await self._run_turn(
            episode,
            role="dungeon_master",
            tools=DM_TOOLS,
        )
        episode.step_count += 1
        return episode, episode.step_count >= self.max_user_responses

    def format_episode(self, episode: RPGEpisode) -> dict:
        """Convert a finished RPGEpisode to a dictionary for storage."""
        return {
            "step_count": episode.step_count,
            "game_setting": episode.game_setting,
            "player_character": episode.player_character,
            "actions": [
                {"role": action.role, "messages": action.messages}
                for action in episode.actions
            ],
            "metrics": episode.metrics,
            "metadata": episode.metadata,
        }

    @staticmethod
    def _dm_tool_use_fewshot() -> Conversation:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I rush the goblin and swing my sword."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "fewshot_roll_attack",
                        "name": "roll_dice",
                        "input": {"notation": "1d20", "reason": "attack roll"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "fewshot_roll_attack",
                        "content": '{"notation":"1d20","reason":"attack roll","rolls":[7],"modifier":0,"total":7}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Your blade whistles past the goblin, and it snaps back with a vicious counterstrike.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I sprint for the collapsing bridge and leap the gap.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "fewshot_roll_athletics",
                        "name": "roll_dice",
                        "input": {"notation": "1d20", "reason": "jump check"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "fewshot_roll_athletics",
                        "content": '{"notation":"1d20","reason":"jump check","rolls":[18],"modifier":0,"total":18}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "You clear the gap and crash onto the far ledge as stone crumbles into the abyss behind you.",
                    }
                ],
            },
        ]

    async def _generate_setting_and_characters(
        self, theme_input: str
    ) -> tuple[str, str]:
        """Generate game parameters (setting and characters) using XML parsing."""
        param_tool: ToolParam = {
            "name": "set_game_parameters",
            "description": "Set the game setting and player character for the roleplaying scenario.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "game_setting": {
                        "type": "string",
                        "description": "Concise setting for the adventure (e.g., 'abandoned library', 'misty forest').",
                    },
                    "player_character": {
                        "type": "string",
                        "description": "Concise description of the player character (e.g., 'rogue archaeologist').",
                    },
                },
                "required": ["game_setting", "player_character"],
            },
        }
        system_text_block: TextBlock = {"type": "text", "text": game_parameter_prompt()}
        user_text_block: TextBlock = {
            "type": "text",
            "text": f"Generate game parameters for a roleplaying scenario based on this theme: {theme_input}",
        }
        strict_user_block: TextBlock = {
            "type": "text",
            "text": (
                "IMPORTANT: Return ONLY the XML tags below and nothing else.\n"
                "<game_setting>...</game_setting>\n"
                "<player_character>...</player_character>"
            ),
        }
        base_conversation: Conversation = [
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
            parameter_conversation = list(base_conversation)
            if attempt > 0:
                parameter_conversation.append(
                    {"role": "user", "content": [strict_user_block]}
                )
            args = GenerationArgs(
                n_retries=2,
                tools=[param_tool],
                tool_choice={"type": "tool", "name": "set_game_parameters"},
            )
            result = await self.generation_wrappers["adventure_parameters"].generate(
                parameter_conversation,
                args=args,
            )
            try:
                # Get the assistant message from the conversation (last message added by generate)
                assistant_message = result.conversation[-1]
                content_blocks = assistant_message.get("content", [])
                for block in content_blocks:
                    if (
                        block.get("type") == "tool_use"
                        and block.get("name") == "set_game_parameters"
                    ):
                        tool_input = block.get("input", {})
                        if isinstance(tool_input, str):
                            try:
                                tool_input = json.loads(tool_input)
                            except json.JSONDecodeError:
                                tool_input = {}
                        game_setting = tool_input.get("game_setting")
                        player_character = tool_input.get("player_character")
                        if game_setting and player_character:
                            return game_setting, player_character
            except ValueError as e:
                if attempt < MAX_PARSE_RETRIES - 1:
                    logger.warning(
                        f"Parse failed (attempt {attempt + 1}), retrying: {e}"
                    )
                else:
                    logger.warning(
                        f"Using fallback game parameters after {MAX_PARSE_RETRIES} attempts"
                    )
        raise ValueError("Failed to generate game parameters")

    @staticmethod
    def _convert_tool_calls_to_text_blocks(
        content_blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Since user messages can't contain tool calls, convert them to text."""
        out_blocks: list[ContentBlock] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_call_json_str = json.dumps(
                    {"name": block.get("name"), "input": block.get("input")}
                )
                out_blocks.append(
                    {
                        "type": "text",
                        "text": f"<user_tool_call>{tool_call_json_str}</user_tool_call>",
                    }
                )
                continue
            if block_type == "tool_result":
                out_blocks.append(
                    {
                        "type": "text",
                        "text": f"<tool_result>{block.get('content', '')}</tool_result>",
                    }
                )
                continue
            out_blocks.append(block)
        return out_blocks

    @staticmethod
    def _has_tool_use_block(message: Message) -> bool:
        content = message.get("content", [])
        if not isinstance(content, list):
            return False
        return any(block.get("type") == "tool_use" for block in content)

    @staticmethod
    def _contains_textual_roll(message: Message) -> bool:
        content = message.get("content", [])
        if not isinstance(content, list):
            return False
        text_parts = [
            str(block.get("text", ""))
            for block in content
            if block.get("type") == "text" and block.get("text")
        ]
        if not text_parts:
            return False
        return DICE_OR_ROLL_TEXT_RE.search("\n".join(text_parts)) is not None

    async def _run_turn(
        self,
        episode: RPGEpisode,
        role: RolePlayingRole,
        tools: list[dict],
        tool_choice: dict | None = None,
    ) -> None:
        conversation = self._format_conversation(episode, role)
        # Keep tool-calling bounded to avoid runaway latency in multi-turn episodes.
        max_tool_iterations = 1 if role == "player" else 2
        args = GenerationArgs(
            n_retries=1,
            max_tokens=TURN_MAX_TOKENS_BY_ROLE[role],
            max_tool_iterations=max_tool_iterations,
            tools=tools or None,
            tool_use_executor=execute_tool_use_block if tools else None,
            tool_choice=tool_choice,
        )
        result = await self.generation_wrappers[role].generate(conversation, args=args)
        episode.actions.append(Action(role=role, messages=result.added_messages))

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
        if generating_role == "dungeon_master":
            conversation.extend(self._dm_tool_use_fewshot())

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

            content = action.message.get("content", [])

            # Convert tool_use and tool_result blocks when role is "user"
            # since user messages can't contain tool_use blocks in the Anthropic API
            if role == "user":
                content = self._convert_tool_calls_to_text_blocks(content)
            else:
                # Preserve trajectory details safely by serializing tool blocks as text
                # rather than replaying provider-native tool payloads.
                content = self._convert_tool_calls_to_text_blocks(content)

            # Skip messages with empty content (except final assistant message)
            if not content:
                continue

            message: Message = {
                "role": role,
                "content": content,
            }

            conversation.append(message)

        return conversation
