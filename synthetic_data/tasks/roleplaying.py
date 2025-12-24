import json
import random
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from jinja2 import Template
from loguru import logger
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from synthetic_data.generation import (
    GenerationArgs,
    GenerationResult,
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
    PLAYER_TOOLS,
    ToolResult,
    execute_tool_calls,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    log_conversation,
    parse_xml_tags,
)

MAX_PARSE_RETRIES = 3
MAX_TOOL_ITERATIONS = 5


@dataclass
class RPGEpisode:
    step_count: int = 0
    game_setting: str | None = None
    player_character: str | None = None
    input_prompt: str | None = None
    parameters_generated: bool = False
    scenario_generated: bool = False
    conversation: Conversation = field(default_factory=list)
    tool_calls_history: list[dict[str, Any]] = field(default_factory=list)
    run_metadata: dict = field(default_factory=dict)
    seed: int = field(default_factory=lambda: random.randint(0, 2**32))


USE_DEV_MODELS = True


def format_tool_result_for_conversation(result: ToolResult) -> dict[str, Any]:
    """Format a tool result for inclusion in the conversation."""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": json.dumps(result.content) if result.success else result.error,
    }


def format_tool_calls_for_conversation(
    tool_calls: list[ChatCompletionMessageToolCall],
) -> list[dict[str, Any]]:
    """Format tool calls for inclusion in assistant message."""
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


async def generate_with_tools_loop(
    wrapper: GenerationWrapper,
    conversation: Conversation,
    tools: list[dict[str, Any]],
    args: GenerationArgs | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> tuple[GenerationResult, list[ToolResult]]:
    """
    Generate with tools, executing tool calls and continuing until the model stops.
    Returns the final generation result and all tool results from the loop.
    """
    current_conversation = list(conversation)
    all_tool_results: list[ToolResult] = []

    gen_args = args or GenerationArgs()
    gen_args = gen_args.model_copy(update={"tools": tools})

    result: GenerationResult | None = None

    for iteration in range(max_iterations):
        results = await wrapper.generate_with_tools([current_conversation], gen_args)
        if not results:
            raise ValueError("No generation results returned")

        result = results[0]

        if not result.tool_calls:
            return result, all_tool_results

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": result.content,
            "tool_calls": format_tool_calls_for_conversation(result.tool_calls),
        }
        current_conversation.append(assistant_message)  # type: ignore

        tool_results = execute_tool_calls(result.tool_calls)
        all_tool_results.extend(tool_results)

        for tool_result in tool_results:
            current_conversation.append(format_tool_result_for_conversation(tool_result))  # type: ignore

        logger.debug(f"Iteration {iteration + 1}: executed {len(result.tool_calls)} tool calls")

    logger.warning(f"Reached max iterations ({max_iterations}) in tool loop")
    if result is None:
        raise ValueError("No generation result after tool loop")
    return result, all_tool_results


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

        self._add_generation_wrapper("generation", gen_model)
        self._add_generation_wrapper("followup", gen_model)
        self._add_generation_wrapper("parameter", gen_model)

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic for roleplaying game.
        Since this is a synthetic task, we create dummy data.
        """
        logger.info("Loading custom dataset for roleplaying game")
        dummy_data = [
            {
                "input_prompt": "A mysterious forest adventure",
                "seed": random.randint(0, 2**32),
            }
            for _ in range(self.num_episodes)
        ]
        logger.info(f"Created {len(dummy_data)} episodes")
        return Dataset.from_list(dummy_data)

    async def start_episode(self, sample: None) -> RPGEpisode:
        seed = hash(str(sample)) % (2**32)
        ep = RPGEpisode()
        ep.run_metadata = {
            "generation_model": self.generation_wrappers["generation"].provider_name,
            "followup_model": self.generation_wrappers["followup"].provider_name,
            "parameter_model": self.generation_wrappers["parameter"].provider_name,
            "max_user_responses": self.max_user_responses,
            "seed": seed,
            "parameters_generated": False,
            "scenario_generated": False,
            "user_responses_generated": False,
        }
        return ep

    async def step_episode(self, episode: RPGEpisode) -> RPGEpisode | None:
        if not episode.parameters_generated:
            await self._generate_parameters(episode)
            episode.parameters_generated = True
            return None

        if not episode.scenario_generated:
            await self._generate_scenario(
                episode, self.generation_wrappers["generation"]
            )
            episode.scenario_generated = True
            return None

        if episode.step_count < self.max_user_responses:
            success = await self._generate_turn(
                episode, self.generation_wrappers["followup"]
            )
            if not success:
                if self._has_user_action(episode):
                    logger.warning(
                        f"Turn generation failed at step {episode.step_count + 1}, "
                        f"finishing episode with {episode.step_count} turns"
                    )
                    return episode
                else:
                    raise ValueError(
                        "Failed to generate turn before any user actions were collected"
                    )
            episode.step_count += 1
            return None

        return episode

    async def _generate_parameters(self, episode: RPGEpisode):
        """Generate game parameters (setting and characters) using XML parsing."""
        parameter_conversation: Conversation = [
            {
                "role": "system",
                "content": GAME_PARAMETER_PROMPT,
            },
            {
                "role": "user",
                "content": f"Generate game parameters for a roleplaying scenario based on this theme: {episode.input_prompt or 'A mysterious adventure'}",
            },
        ]

        logger.info(
            f"Generating game parameters with {self.generation_model_names['parameter']}"
        )

        for attempt in range(MAX_PARSE_RETRIES):
            response = await self.generation_wrappers["parameter"].generate(
                [parameter_conversation]
            )
            try:
                parsed_tags = parse_xml_tags(
                    response[0], required_tags=["game_setting", "player_character"]
                )
                episode.game_setting = parsed_tags["game_setting"]
                episode.player_character = parsed_tags["player_character"]
                logger.debug(f"\nGame setting:\n{episode.game_setting}")
                logger.debug(f"\nPlayer character:\n{episode.player_character}")
                return
            except ValueError as e:
                if attempt < MAX_PARSE_RETRIES - 1:
                    logger.warning(f"Parse failed (attempt {attempt + 1}), retrying: {e}")
                else:
                    raise ValueError(f"Failed to generate game parameters after {MAX_PARSE_RETRIES} attempts")

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
        """Flip user and assistant roles in a conversation while preserving system messages."""
        flipped_conversation = []
        for message in conversation:
            if message["role"] == "user":
                flipped_conversation.append(
                    {"role": "assistant", "content": message["content"]}  # type: ignore
                )
            elif message["role"] == "assistant":
                new_msg: dict[str, Any] = {"role": "user", "content": message["content"]}
                if "tool_calls" in message:
                    new_msg["tool_calls"] = message["tool_calls"]
                flipped_conversation.append(new_msg)  # type: ignore
            elif message["role"] == "tool":
                flipped_conversation.append(message)
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

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": result.content,
        }
        if result.tool_calls:
            assistant_message["tool_calls"] = format_tool_calls_for_conversation(result.tool_calls)

        episode.conversation.append(assistant_message)  # type: ignore

        for tool_result in tool_results:
            episode.tool_calls_history.append({
                "turn": 0,
                "tool_call_id": tool_result.tool_call_id,
                "result": tool_result.content,
                "success": tool_result.success,
            })

        logger.debug(f"Scenario generated with {len(tool_results)} tool calls")

    def _has_user_action(self, episode: RPGEpisode) -> bool:
        """Check if the episode has at least one user action."""
        return any(m["role"] == "user" for m in episode.conversation)

    async def _generate_turn(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ) -> bool:
        """
        Generate a single turn of conversation (player action + DM response).
        Returns True if the turn was generated successfully, False if generation failed.
        """
        flipped_conversation = self._flip_conversation_roles(episode.conversation)

        user_action_conversation: Conversation = [
            {
                "role": "system",
                "content": self._user_action_system_prompt(episode),
            },
            *flipped_conversation,
        ]
        log_conversation(user_action_conversation)

        logger.info(
            f"Generating user response {episode.step_count + 1} with {self.generation_model_names['followup']}"
        )

        try:
            user_result, user_tool_results = await generate_with_tools_loop(
                self.generation_wrappers["followup"],
                user_action_conversation,
                PLAYER_TOOLS,
            )
        except Exception as e:
            logger.warning(f"Failed to generate user response: {e}")
            return False

        user_message: dict[str, Any] = {
            "role": "user",
            "content": user_result.content,
        }
        if user_result.tool_calls:
            user_message["tool_calls"] = format_tool_calls_for_conversation(user_result.tool_calls)

        episode.conversation.append(user_message)  # type: ignore

        for tool_result in user_tool_results:
            episode.tool_calls_history.append({
                "turn": episode.step_count + 1,
                "role": "player",
                "tool_call_id": tool_result.tool_call_id,
                "result": tool_result.content,
                "success": tool_result.success,
            })

        dm_conversation: Conversation = [
            {
                "role": "system",
                "content": self._game_master_system_prompt(episode),
            },
            *episode.conversation,
        ]

        log_conversation(dm_conversation)

        logger.info(
            f"Generating DM response {episode.step_count + 1} with {self.generation_model_names['generation']}"
        )

        try:
            dm_result, dm_tool_results = await generate_with_tools_loop(
                generation_wrapper,
                dm_conversation,
                DM_TOOLS,
            )
        except Exception as e:
            logger.warning(f"Failed to generate DM response: {e}")
            return False

        dm_message: dict[str, Any] = {
            "role": "assistant",
            "content": dm_result.content,
        }
        if dm_result.tool_calls:
            dm_message["tool_calls"] = format_tool_calls_for_conversation(dm_result.tool_calls)

        episode.conversation.append(dm_message)  # type: ignore

        for tool_result in dm_tool_results:
            episode.tool_calls_history.append({
                "turn": episode.step_count + 1,
                "role": "dm",
                "tool_call_id": tool_result.tool_call_id,
                "result": tool_result.content,
                "success": tool_result.success,
            })

        logger.info(f"Generated turn {episode.step_count + 1}")
        return True

    def get_output_row(self, episode: RPGEpisode) -> list[dict]:
        tool_calls_history = episode.tool_calls_history if episode.tool_calls_history else None
        return [
            {
                "conversation": episode.conversation,
                "game_setting": episode.game_setting or "No setting generated",
                "player_character": episode.player_character or "No characters generated",
                "generation_model": self.generation_model_names["generation"],
                "followup_model": self.generation_model_names["followup"],
                "parameter_model": self.generation_model_names["parameter"],
                "num_turns": episode.step_count,
                "tool_calls_history": tool_calls_history,
            }
        ]
