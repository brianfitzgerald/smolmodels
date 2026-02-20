import asyncio
import json
import random
import re
import time
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
TIMEOUT_BUFFER_S = 5.0
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
RISKY_ACTION_TEXT_RE = re.compile(
    r"\b(attack|strike|swing|shoot|fight|dodge|parry|block|jump|climb|sneak|stealth|hide|pick|lock|search|perception|listen|rush|charge|run)\b",
    re.IGNORECASE,
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
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "turn_count": 0,
            "total_tool_use_blocks": 0,
            "dm_tool_use_blocks": 0,
            "player_tool_use_blocks": 0,
        }
    )
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
        conversation_lines = self._flatten_action_messages(episode.actions)
        return {
            "step_count": episode.step_count,
            "game_setting": episode.game_setting,
            "player_character": episode.player_character,
            "actions": [
                {"role": action.role, "messages": action.messages}
                for action in episode.actions
            ],
            "conversation_lines": conversation_lines,
            "transcript": "\n".join(conversation_lines),
            "metrics": episode.metrics
            if episode.metrics
            else {
                "turn_count": 0,
                "total_tool_use_blocks": 0,
                "dm_tool_use_blocks": 0,
                "player_tool_use_blocks": 0,
            },
            "metadata": episode.metadata,
        }

    @staticmethod
    def _flatten_action_messages(actions: list[Action]) -> list[str]:
        lines: list[str] = []
        for action in actions:
            for message in action.messages:
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                chunks: list[str] = []
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text = str(block.get("text", "")).strip()
                        if text:
                            chunks.append(text)
                    elif block_type == "tool_use":
                        name = str(block.get("name", "")).strip()
                        payload = block.get("input", {})
                        chunks.append(
                            f"<tool_use name={name}>{json.dumps(payload, ensure_ascii=False)}</tool_use>"
                        )
                    elif block_type == "tool_result":
                        result = str(block.get("content", "")).strip()
                        chunks.append(f"<tool_result>{result}</tool_result>")
                if chunks:
                    lines.append(f"{action.role}: {' '.join(chunks)}")
        return lines

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
            result = await self._generate_with_latency_guard(
                "adventure_parameters",
                parameter_conversation,
                args,
                stage=f"setting_generation_attempt_{attempt + 1}",
            )
            try:
                # Get the assistant message from the conversation (last message added by generate)
                if not result.conversation:
                    raise ValueError("No conversation returned by generation wrapper")
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
                raise ValueError("Missing set_game_parameters tool output")
            except Exception as e:
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

    @staticmethod
    def _contains_risky_action_request(message: Message) -> bool:
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
        return RISKY_ACTION_TEXT_RE.search("\n".join(text_parts)) is not None

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
        result = await self._generate_with_latency_guard(
            role,
            conversation,
            args,
            stage=f"{role}_turn_{episode.step_count}",
        )
        if role == "dungeon_master" and tools:
            latest_user = next(
                (m for m in reversed(conversation) if m.get("role") == "user"),
                None,
            )
            latest_assistant = next(
                (
                    m
                    for m in reversed(result.added_messages)
                    if m.get("role") == "assistant"
                ),
                None,
            )
            if (
                latest_user is not None
                and self._contains_risky_action_request(latest_user)
                and latest_assistant is not None
                and not self._has_tool_use_block(latest_assistant)
            ):
                logger.warning(
                    "DM resolved risky action without a tool call; retrying with forced tool usage"
                )
                retry_args = args.model_copy(deep=True)
                retry_args.tool_choice = {"type": "any"}
                retry_result = await self._generate_with_latency_guard(
                    role,
                    conversation,
                    retry_args,
                    stage=f"{role}_turn_{episode.step_count}_forced_risky_tool_retry",
                )
                retry_latest_assistant = next(
                    (
                        m
                        for m in reversed(retry_result.added_messages)
                        if m.get("role") == "assistant"
                    ),
                    None,
                )
                if retry_latest_assistant is not None and self._has_tool_use_block(
                    retry_latest_assistant
                ):
                    result = retry_result
            elif (
                latest_assistant is not None
                and self._contains_textual_roll(latest_assistant)
                and not self._has_tool_use_block(latest_assistant)
            ):
                logger.warning(
                    "DM narrated dice/randomness without a tool call; retrying with forced tool usage"
                )
                retry_args = args.model_copy(deep=True)
                retry_args.tool_choice = {"type": "any"}
                retry_result = await self._generate_with_latency_guard(
                    role,
                    conversation,
                    retry_args,
                    stage=f"{role}_turn_{episode.step_count}_forced_tool_retry",
                )
                retry_latest_assistant = next(
                    (
                        m
                        for m in reversed(retry_result.added_messages)
                        if m.get("role") == "assistant"
                    ),
                    None,
                )
                if retry_latest_assistant is not None and self._has_tool_use_block(
                    retry_latest_assistant
                ):
                    result = retry_result
        tool_use_blocks = self._count_tool_use_blocks(result.added_messages)
        episode.metrics["turn_count"] = int(episode.metrics.get("turn_count", 0)) + 1
        episode.metrics["total_tool_use_blocks"] = (
            int(episode.metrics.get("total_tool_use_blocks", 0)) + tool_use_blocks
        )
        key = (
            "dm_tool_use_blocks"
            if role == "dungeon_master"
            else "player_tool_use_blocks"
        )
        episode.metrics[key] = int(episode.metrics.get(key, 0)) + tool_use_blocks
        episode.actions.append(Action(role=role, messages=result.added_messages))

    @staticmethod
    def _count_tool_use_blocks(messages: list[Message]) -> int:
        count = 0
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    count += 1
        return count

    async def _generate_with_latency_guard(
        self,
        role: Literal["dungeon_master", "player", "adventure_parameters"],
        conversation: Conversation,
        args: GenerationArgs,
        stage: str,
    ):
        wrapper = self.generation_wrappers[role]
        timeout_s = wrapper.gen_wrapper_args.request_timeout_s + TIMEOUT_BUFFER_S
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                wrapper.generate(conversation, args=args),
                timeout=timeout_s,
            )
            elapsed_s = time.perf_counter() - start
            logger.info(f"{stage} completed in {elapsed_s:.2f}s")
            return result
        except asyncio.TimeoutError as e:
            elapsed_s = time.perf_counter() - start
            logger.error(
                f"{stage} timed out after {elapsed_s:.2f}s (timeout={timeout_s:.1f}s)"
            )
            raise TimeoutError(f"{stage} timed out after {elapsed_s:.2f}s") from e

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

            # Flatten all content blocks from this turn so we keep trajectory detail,
            # then normalize tool blocks to text tags for safe replay.
            merged_blocks: list[ContentBlock] = []
            for turn_message in action.messages:
                msg_content = turn_message.get("content", [])
                if isinstance(msg_content, list):
                    merged_blocks.extend(msg_content)
            content = self._convert_tool_calls_to_text_blocks(merged_blocks)

            if not content:
                continue

            message: Message = {
                "role": role,
                "content": content,
            }

            conversation.append(message)

        return conversation
