import random
import re
from typing import TypedDict

from datasets import Dataset
from jinja2 import Template
from loguru import logger
from pydantic import BaseModel, Field

from synthetic_data.generation import GenerationArgs, GenerationWrapper, RemoteModel
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    GAME_PARAMETER_PROMPT,
    ROLEPLAYING_PROMPT,
    USER_ACTION_PROMPT,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    log_conversation,
    parse_xml_tags,
)

MAX_PARSE_RETRIES = 3

# Dice roll pattern: matches "1d20", "2d6+3", "d20", "3d8-2", etc.
DICE_PATTERN = re.compile(r"\b(\d*)d(\d+)([+-]\d+)?\b", re.IGNORECASE)


def roll_dice(
    num_dice: int, num_sides: int, modifier: int = 0
) -> tuple[list[int], int]:
    """Roll dice and return individual rolls and total."""
    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    total = sum(rolls) + modifier
    return rolls, total


def parse_and_roll_dice(text: str) -> list[dict]:
    """Find all dice notation in text and roll them.

    Returns a list of roll results with notation, rolls, and total.
    """
    results = []
    for match in DICE_PATTERN.finditer(text):
        num_dice = int(match.group(1)) if match.group(1) else 1
        num_sides = int(match.group(2))
        modifier = int(match.group(3)) if match.group(3) else 0

        rolls, total = roll_dice(num_dice, num_sides, modifier)

        notation = f"{num_dice}d{num_sides}"
        if modifier > 0:
            notation += f"+{modifier}"
        elif modifier < 0:
            notation += str(modifier)

        results.append(
            {
                "notation": notation,
                "rolls": rolls,
                "modifier": modifier,
                "total": total,
            }
        )
    return results


def format_dice_results(results: list[dict]) -> str:
    """Format dice roll results as a string."""
    if not results:
        return ""

    parts = []
    for r in results:
        rolls_str = ", ".join(str(x) for x in r["rolls"])
        if r["modifier"] != 0:
            mod_str = (
                f" + {r['modifier']}"
                if r["modifier"] > 0
                else f" - {abs(r['modifier'])}"
            )
            parts.append(
                f"[{r['notation']}: rolled {rolls_str}{mod_str} = {r['total']}]"
            )
        else:
            if len(r["rolls"]) == 1:
                parts.append(f"[{r['notation']}: {r['total']}]")
            else:
                parts.append(f"[{r['notation']}: rolled {rolls_str} = {r['total']}]")

    return "\n".join(parts)


async def generate_with_retry(
    wrapper: GenerationWrapper,
    conversations: list[Conversation],
    required_tags: list[str],
    args: GenerationArgs | None = None,
) -> dict[str, str] | None:
    """Generate and parse XML tags, retrying on parse failure."""
    for attempt in range(MAX_PARSE_RETRIES):
        response = await wrapper.generate(conversations, args)
        try:
            return parse_xml_tags(response[0], required_tags=required_tags)
        except ValueError as e:
            if attempt < MAX_PARSE_RETRIES - 1:
                logger.warning(f"Parse failed (attempt {attempt + 1}), retrying: {e}")
            else:
                logger.error(f"Parse failed after {MAX_PARSE_RETRIES} attempts")
                return None


class ScenarioTags(TypedDict, total=False):
    game_design: str
    dm_narration: str
    dm_response: str
    npc_dialogue: str | None


class RPGEpisode(BaseModel):
    step_count: int = 0
    game_setting: str | None = None
    player_character: str | None = None
    scenario: str | None = None
    scenario_tags: ScenarioTags | None = None
    parameters_generated: bool = False
    scenario_generated: bool = False
    conversation: Conversation = Field(default_factory=list)
    dice_rolls: list[dict] = Field(default_factory=list)  # Track all dice rolls
    run_metadata: dict = Field(default_factory=dict)
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32))


USE_DEV_MODELS = True


class RoleplayingGameMultiStepTask(BaseTask[None, RPGEpisode]):
    """
    Multi-step variant:
    Step 1: Generate game parameters (setting and characters)
    Step 2: Generate the roleplaying scenario using ROLEPLAYING_PROMPT
    Step 3: Generate simulated user responses
    """

    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_name = "roleplaying_game_multi_step"
    output_dataset_org = "roborovski"
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(
        self,
        run_mode: RunMode,
        max_user_responses: int = 10,
        input_prompt: str = "A mysterious forest adventure",
        num_episodes: int = 100,
    ):
        super().__init__(run_mode)
        self.max_user_responses = max_user_responses
        self.input_prompt = input_prompt
        self.num_episodes = num_episodes

        gen_model: RemoteModel = (
            "claude-3-5-haiku" if USE_DEV_MODELS else "claude-4-5-sonnet"
        )

        self._add_generation_wrapper("generation", gen_model)
        self._add_generation_wrapper("followup", gen_model)
        self._add_generation_wrapper("parameter", gen_model)

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic for roleplaying game.
        Since this is a synthetic task, we create dummy data.
        """
        logger.info(
            f"Loading custom dataset for roleplaying game with prompt: {self.input_prompt}"
        )
        # Create dummy data for the roleplaying game
        dummy_data = [{"prompt": self.input_prompt} for _ in range(self.num_episodes)]
        logger.info(f"Created {len(dummy_data)} episodes")
        return Dataset.from_list(dummy_data)

    async def start_episode(self, sample: None) -> RPGEpisode:
        seed = hash(str(sample)) % (2**32)
        # Update wrappers with seed for deterministic generation
        ep = RPGEpisode()
        ep.run_metadata = {
            "generation_model": self.generation_wrappers["generation"].provider_name,
            "followup_model": self.generation_wrappers["followup"].provider_name,
            "parameter_model": self.generation_wrappers["parameter"].provider_name,
            "max_user_responses": self.max_user_responses,
            "input_prompt": self.input_prompt,
            "seed": seed,
            "parameters_generated": False,
            "scenario_generated": False,
            "user_responses_generated": False,
        }
        logger.info(f"Start episode with prompt: {self.input_prompt}, seed: {seed}")
        return ep

    async def step_episode(self, episode: RPGEpisode) -> RPGEpisode | None:
        # Step 0: Generate game parameters and scenario if not done yet
        if not episode.parameters_generated:
            await self._generate_parameters(episode)
            episode.parameters_generated = True
            return None  # Still in progress

        if not episode.scenario_generated:
            await self._generate_scenario(
                episode, self.generation_wrappers["generation"]
            )
            episode.scenario_generated = True
            return None  # Still in progress

        # Step 1+: Generate turn-by-turn conversation
        if episode.step_count < self.max_user_responses:
            success = await self._generate_turn(
                episode, self.generation_wrappers["followup"]
            )
            if not success:
                # If we have at least 1 user action, finish the sample with what we have
                if self._has_user_action(episode):
                    logger.warning(
                        f"Turn generation failed at step {episode.step_count + 1}, "
                        f"finishing episode with {episode.step_count} turns"
                    )
                    return episode  # Return partial episode
                else:
                    raise ValueError(
                        "Failed to generate turn before any user actions were collected"
                    )
            episode.step_count += 1
            return None  # Still in progress

        return episode  # Episode complete

    async def _generate_parameters(self, episode: RPGEpisode):
        """Generate game parameters (setting and characters)"""
        parameter_conversation: Conversation = [
            {
                "role": "system",
                "content": GAME_PARAMETER_PROMPT
                + "\n\nYou are tasked with generating game parameters for a roleplaying scenario. Create engaging and detailed content.",
            },
            {
                "role": "user",
                "content": f"Generate game parameters for a roleplaying scenario based on this theme: {self.input_prompt}\n\nPlease provide:\n1. A detailed GAME_SETTING (2-3 paragraphs)\n2. PLAYER_CHARACTER information (1-2 paragraphs describing the main character)",
            },
        ]

        logger.info(
            f"Generating game parameters with {self.generation_model_names['parameter']}"
        )
        parsed_tags = await generate_with_retry(
            self.generation_wrappers["parameter"],
            [parameter_conversation],
            required_tags=["game_setting", "player_character"],
        )
        if parsed_tags is None:
            raise ValueError("Failed to generate game parameters")
        episode.game_setting = parsed_tags["game_setting"]
        episode.player_character = parsed_tags["player_character"]

        logger.debug(f"\nGame setting:\n{episode.game_setting}")
        logger.debug(f"\nPlayer character:\n{episode.player_character}")

    def _game_master_system_prompt(self, episode: RPGEpisode) -> str:
        template = Template(ROLEPLAYING_PROMPT)
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
                flipped_conversation.append(
                    {"role": "user", "content": message["content"]}  # type: ignore
                )
            else:
                # Keep system messages as-is
                flipped_conversation.append(message)
        return flipped_conversation

    async def _generate_scenario(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
        """Generate the initial roleplaying scenario"""

        scenario_conversation: Conversation = [
            {
                "role": "user",
                "content": self._game_master_system_prompt(episode),
            },
        ]

        logger.info(
            f"Generating scenario with {self.generation_model_names['generation']}"
        )
        parsed_tags = await generate_with_retry(
            generation_wrapper,
            [scenario_conversation],
            required_tags=["game_design", "dm_narration"],
            args=GenerationArgs(
                max_tokens=8192, temperature=1, prefill="<game_design>"
            ),
        )
        if parsed_tags is None:
            raise ValueError("Failed to generate scenario")
        episode.scenario_tags = ScenarioTags(**parsed_tags)
        opening_message = parsed_tags["dm_narration"]
        if "npc_dialogue" in parsed_tags:
            opening_message += f"\n\n{parsed_tags['npc_dialogue']}"

        # Check for dice rolls in the opening narration
        dice_results = parse_and_roll_dice(opening_message)
        if dice_results:
            for result in dice_results:
                episode.dice_rolls.append(
                    {
                        "turn": 0,
                        **result,
                    }
                )
            dice_str = format_dice_results(dice_results)
            logger.info(f"Dice rolled in opening: {dice_str}")
            opening_message += f"\n\n{dice_str}"

        episode.conversation.append(
            {
                "role": "assistant",
                "content": opening_message,
            }
        )
        logger.debug(f"Scenario:\n{episode.scenario_tags}")

    def _has_user_action(self, episode: RPGEpisode) -> bool:
        """Check if the episode has at least one user action."""
        return any(m["role"] == "user" for m in episode.conversation)

    async def _generate_turn(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ) -> bool:
        """Generate a single turn of conversation (user response + assistant response).

        Returns True if the turn was generated successfully, False if generation failed.
        """

        assert episode.scenario_tags is not None

        # Flip roles in the conversation: user becomes assistant and vice versa
        flipped_conversation = self._flip_conversation_roles(episode.conversation)

        template = Template(USER_ACTION_PROMPT)
        formatted_user_action_prompt = template.render(
            GAME_SETTING=episode.game_setting,
            PLAYER_CHARACTER=episode.player_character,
        )

        user_action_conversation: Conversation = [
            {
                "role": "system",
                "content": formatted_user_action_prompt,
            },
            *flipped_conversation,
        ]
        log_conversation(user_action_conversation)

        logger.info(
            f"Generating user response {episode.step_count + 1} with {self.generation_model_names['followup']}"
        )
        user_response_tags = await generate_with_retry(
            self.generation_wrappers["followup"],
            [user_action_conversation],
            required_tags=["user_action"],
        )
        if user_response_tags is None:
            logger.warning("Failed to generate user response")
            return False
        user_action_content = user_response_tags["user_action"]

        episode.conversation.append({"role": "user", "content": user_action_content})

        # Generate assistant response
        assistant_conversation: Conversation = [
            {
                "role": "system",
                "content": self._game_master_system_prompt(episode),
            },
            {
                "role": "user",
                "content": f"Game Setting: {episode.game_setting}\nPlayer Character: {episode.player_character}\nPlayer Response: {user_action_content}\n\nRespond as the dungeon master:",
            },
            *episode.conversation,
        ]

        log_conversation(assistant_conversation)

        logger.info(
            f"Generating assistant response {episode.step_count + 1} with {self.generation_model_names['generation']}"
        )
        parsed_tags = await generate_with_retry(
            generation_wrapper,
            [assistant_conversation],
            required_tags=["dm_narration"],
        )
        if parsed_tags is None:
            logger.warning("Failed to generate assistant response")
            return False
        formatted_dm_response = parsed_tags["dm_narration"]

        # Check for dice rolls in the DM response
        dice_results = parse_and_roll_dice(formatted_dm_response)
        if dice_results:
            # Track the rolls
            for result in dice_results:
                episode.dice_rolls.append(
                    {
                        "turn": episode.step_count + 1,
                        **result,
                    }
                )
            # Format and append dice results to the response
            dice_str = format_dice_results(dice_results)
            logger.info(f"Dice rolled: {dice_str}")
            formatted_dm_response += f"\n\n{dice_str}"

        episode.conversation.append(
            {"role": "assistant", "content": formatted_dm_response}
        )

        logger.info(f"Generated turn {episode.step_count + 1}")
        return True

    def get_output_row(self, episode: RPGEpisode) -> list[dict]:
        return [
            {
                "conversation": episode.conversation,
                "game_setting": episode.game_setting or "No setting generated",
                "player_character": episode.player_character
                or "No characters generated",
                "original_input": {"prompt": self.input_prompt},
                "generation_model": self.generation_model_names["generation"],
                "followup_model": self.generation_model_names["followup"],
                "parameter_model": self.generation_model_names["parameter"],
                "num_turns": episode.step_count,
                "dice_rolls": episode.dice_rolls,
            }
        ]
