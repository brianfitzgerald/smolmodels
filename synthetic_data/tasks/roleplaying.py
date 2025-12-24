import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Dataset
from jinja2 import Template
from loguru import logger
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from synthetic_data.generation import (
    GenerationArgs,
    GenerationWrapper,
    RemoteModel,
)
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    GAME_PARAMETER_PROMPT,
    ROLEPLAYING_PROMPT,
    USER_ACTION_PROMPT,
)
from synthetic_data.tasks.roleplaying_tools import (
    DM_TOOLS,
    ToolResult,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    Message,
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


@dataclass
class Action:
    role: Literal["player", "dm"]
    tool_calls: list[ChatCompletionMessageToolCall]
    content: str


@dataclass
class RPGEpisode:
    step_count: int = 0
    game_setting: str | None = None
    player_character: str | None = None
    input_prompt: str | None = None
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
    output_dataset_name = "roleplaying_game_multi_step"
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

    async def initial_observation(self, sample: None) -> RPGEpisode:
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
        await asyncio.gather(
            self._generate_parameters(ep),
            self._generate_scenario(ep, self.generation_wrappers["dungeon_master"]),
        )
        return ep

    async def step_episode(self, episode: RPGEpisode) -> RPGEpisode | None:
        if episode.step_count < self.max_user_responses:
            success = await self._generate_turn(
                episode, self.generation_wrappers["player"]
            )
            if not success:
                logger.warning(
                    f"Turn generation failed at step {episode.step_count + 1}, "
                    f"finishing episode with {episode.step_count} turns"
                )
                return episode
            episode.step_count += 1
            return None

        return episode

    async def _generate_parameters(self, episode: RPGEpisode):
        """Generate game parameters (setting and characters) using XML parsing."""
        theme_input = episode.input_prompt or "A mysterious adventure"
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
                episode.game_setting = parsed_tags["game_setting"]
                episode.player_character = parsed_tags["player_character"]
                logger.debug(f"\nGame setting:\n{episode.game_setting}")
                logger.debug(f"\nPlayer character:\n{episode.player_character}")
            except ValueError as e:
                if attempt < MAX_PARSE_RETRIES - 1:
                    logger.warning(
                        f"Parse failed (attempt {attempt + 1}), retrying: {e}"
                    )
                else:
                    raise ValueError(
                        f"Failed to generate game parameters after {MAX_PARSE_RETRIES} attempts"
                    )

    def _game_master_system_prompt(self, episode: RPGEpisode) -> str:
        template: Template = Template(ROLEPLAYING_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTER=episode.player_character or "A brave adventurer",
        )
        return formatted_prompt

    def _user_action_system_prompt(self, episode: RPGEpisode) -> str:
        template: Template = Template(USER_ACTION_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTER=episode.player_character or "A brave adventurer",
        )
        return formatted_prompt

    def _flip_conversation_roles(self, conversation: Conversation) -> Conversation:
        """Flip user and assistant roles in a conversation while preserving system messages.

        Note: When flipping assistant to user, we don't copy tool_calls since user messages
        cannot have tool_calls. We also skip tool result messages since they don't make
        sense in the flipped context.
        """
        flipped_conversation = []

        # First pass: collect all tool results
        all_tool_contents: list[str] = []
        for message in conversation:
            if message["role"] == "tool":
                content = message.get("content", "")
                if content:
                    all_tool_contents.append(content)

        # Second pass: process messages
        for i, message in enumerate(conversation):
            if message["role"] == "user":
                flipped_conversation.append(
                    {"role": "assistant", "content": message["content"]}  # type: ignore
                )
            elif message["role"] == "assistant":
                # Don't copy tool_calls to user messages (users can't make tool calls)
                content = message.get("content") or ""

                # If no text content, check for tool results that follow this assistant message
                if not content:
                    # Collect tool results that immediately follow this assistant message
                    following_tool_contents = []
                    for j in range(i + 1, len(conversation)):
                        if conversation[j]["role"] == "tool":
                            tool_content = conversation[j].get("content", "")
                            if tool_content:
                                following_tool_contents.append(tool_content)
                        else:
                            break  # Stop at next non-tool message
                    if following_tool_contents:
                        content = "\n".join(following_tool_contents)

                # Only add if there's meaningful content
                if content:
                    flipped_conversation.append(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )  # type: ignore
            elif message["role"] == "tool":
                # Skip tool result messages - we've extracted their content above
                continue
            else:
                flipped_conversation.append(message)
        return flipped_conversation

    async def _generate_scenario(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
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

        logger.info(
            f"Generating scenario with {self.generation_model_names['generation']}"
        )

        result, tool_results = await generate_with_tools_loop(
            generation_wrapper,
            scenario_conversation,
            DM_TOOLS,
            GenerationArgs(max_tokens=8192, temperature=1),
        )

        episode.conversation.append(assistant_message)  # type: ignore

        # Add tool result messages to conversation (for context when flipping roles)
        for tool_result in tool_results:
            episode.actions.append(
                Action(
                    role="dm",
                    tool_calls=result.tool_calls,
                    content=result.content,
                )
            )

        logger.debug(f"Scenario generated with {len(tool_results)} tool calls")

    def _format_conversation(
        self, episode: RPGEpisode, generating_role: Literal["player", "dm"]
    ) -> Conversation:
        """
        For the DM, generate with player actions as user actions, and DM actions as assistant actions.
        For the player, generate with DM actions as user actions, and player actions as assistant actions.
        """
        conversation: Conversation = []

        for action in episode.actions:
            if generating_role == "dm":
                # DM perspective: player -> user, dm -> assistant
                role = "user" if action.role == "player" else "assistant"
            else:  # generating_role == "player"
                # Player perspective: dm -> user, player -> assistant
                role = "user" if action.role == "dm" else "assistant"

            message: Message = {
                "role": role,
                "content": action.content,
            }

            # Only assistant messages can have tool_calls in the OpenAI API format
            if role == "assistant" and action.tool_calls:
                message["tool_calls"] = action.tool_calls

            conversation.append(message)

        return conversation
