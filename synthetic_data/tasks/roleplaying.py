import random
from jinja2 import Template
from loguru import logger

from synthetic_data.generation import GenerationArgs, GenerationWrapper, RemoteModel
from synthetic_data.tasks import BaseTask, BaseTaskV1, RunMode
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
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import TypedDict


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
    user_responses: list[str] | None = None
    conversation: Conversation = Field(default_factory=list)
    run_metadata: dict = Field(default_factory=dict)
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32))


class RoleplayingGame(BaseTaskV1):
    """
    Generate roleplaying scenarios with follow-up questions using different models.
    """

    output_dataset_name = "roleplaying_scenarios"
    dataset_columns = [
        "game_setting",
        "player_character",
        "scenario",
        "user_responses",
        "original_input",
        "generation_model",
        "followup_model",
        "parameter_model",
    ]
    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_format = DatasetFormat.PARQUET


class RoleplayingGameMultiStepTask(BaseTask[None, RPGEpisode]):
    """
    Multi-step variant:
    Step 1: Generate game parameters (setting and characters)
    Step 2: Generate the roleplaying scenario using ROLEPLAYING_PROMPT
    Step 3: Generate simulated user responses
    """

    seed_data_format = DatasetFormat.CUSTOM

    def __init__(
        self,
        run_mode: RunMode,
        generation_model: RemoteModel = "claude-4-sonnet",
        followup_model: RemoteModel = "claude-4-sonnet",
        parameter_model: RemoteModel = "claude-3-5-haiku",
        max_user_responses: int = 10,
        input_prompt: str = "A mysterious forest adventure",
    ):
        super().__init__(run_mode)
        self.max_user_responses = max_user_responses
        self.input_prompt = input_prompt

        # adopt dataset config from RoleplayingGame single-step task
        self.task = RoleplayingGame(run_mode)
        self.output_dataset_name = self.task.output_dataset_name
        self.output_dataset_org = self.task.output_dataset_org
        self.output_dataset_format = self.task.output_dataset_format
        self.dataset_columns = self.task.dataset_columns

        self._add_generation_wrapper("generation", generation_model)
        self._add_generation_wrapper("followup", followup_model)
        self._add_generation_wrapper("parameter", parameter_model)

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic for roleplaying game.
        Since this is a synthetic task, we create dummy data.
        """
        logger.info(
            f"Loading custom dataset for roleplaying game with prompt: {self.input_prompt}"
        )
        # Create dummy data for the roleplaying game
        dummy_data = [{"prompt": self.input_prompt} for _ in range(10)]  # 10 episodes
        logger.info(f"Created {len(dummy_data)} dummy episodes")
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

    async def step_episode(self, episode: RPGEpisode) -> list[dict]:
        # Step 0: Generate game parameters and scenario if not done yet
        if not episode.parameters_generated:
            await self._generate_parameters(episode)
            episode.parameters_generated = True
            return []

        if not episode.scenario_generated:
            await self._generate_scenario(
                episode, self.generation_wrappers["generation"]
            )
            episode.scenario_generated = True
            return []

        # Step 1+: Generate turn-by-turn conversation
        if episode.step_count < self.max_user_responses:
            await self._generate_turn(episode, self.generation_wrappers["followup"])
            episode.step_count += 1
            return []

        return []

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
        log_conversation(parameter_conversation)
        parameter_response = await self.generation_wrappers["parameter"].generate(
            [parameter_conversation]
        )

        # Parse the response to extract game_setting and player_characters
        response_text = parameter_response[0]

        try:
            parsed_tags = parse_xml_tags(
                response_text, required_tags=["game_setting", "player_character"]
            )
            episode.game_setting = parsed_tags["game_setting"]
            episode.player_character = parsed_tags["player_character"]
        except ValueError as e:
            logger.error(f"Failed to parse required XML tags: {e}")
            raise

        logger.debug(f"\nGame setting:\n{episode.game_setting}")
        logger.debug(f"\nPlayer character:\n{episode.player_character}")

    def _game_master_system_prompt(self, episode: RPGEpisode) -> str:
        template = Template(ROLEPLAYING_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTER=episode.player_character or "A brave adventurer",
        )
        return formatted_prompt

    async def _generate_scenario(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
        """Generate the initial roleplaying scenario"""

        scenario_conversation: Conversation = [
            {
                "role": "user",
                "content": self._game_master_system_prompt(episode),
            },
            {
                "role": "assistant",
                "content": "<game_design>",
            },
        ]
        log_conversation(scenario_conversation)

        logger.info(
            f"Generating scenario with {self.generation_model_names['generation']}"
        )
        scenario_response = await generation_wrapper.generate(
            [scenario_conversation], GenerationArgs(max_tokens=8192, temperature=1)
        )
        # readd prefix
        scenario_response_text = "<game_design>" + scenario_response[0]
        parsed_tags = parse_xml_tags(
            scenario_response_text,
            required_tags=[
                "game_design",
                "dm_narration",
            ],
        )

        # Now that we have metadata, format the conversation with it

        episode.scenario_tags = ScenarioTags(**parsed_tags)
        opening_message = parsed_tags["dm_narration"]
        if "npc_dialogue" in parsed_tags:
            opening_message += f"\n\n{parsed_tags['npc_dialogue']}"
        episode.conversation.append(
            {
                "role": "assistant",
                "content": opening_message,
            }
        )
        logger.debug(f"Scenario:\n{episode.scenario_tags}")

    async def _generate_turn(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
        """Generate a single turn of conversation (user response + assistant response)"""

        assert episode.scenario_tags is not None

        user_action_conversation: Conversation = [
            {
                "role": "system",
                "content": "You are simulating a player in a roleplaying game. Based on the scenario and the dungeon master's response, generate a realistic player response that a human player might make. This should be a natural, in-character response that advances the story or explores the scenario.",
            },
            *episode.conversation,
        ]
        log_conversation(user_action_conversation)

        logger.info(
            f"Generating user response {episode.step_count + 1} with {self.generation_model_names['followup']}"
        )
        user_response_result = await self.generation_wrappers["followup"].generate(
            [user_action_conversation]
        )
        user_response = user_response_result[0]
        logger.debug(f"User response:\n{user_response}")

        episode.conversation.append({"role": "user", "content": user_response})

        # Generate assistant response
        assistant_conversation: Conversation = [
            {
                "role": "system",
                "content": self._game_master_system_prompt(episode),
            },
            {
                "role": "user",
                "content": f"Game Setting: {episode.game_setting}\nPlayer Character: {episode.player_character}\nPlayer Response: {user_response}\n\nRespond as the dungeon master:",
            },
        ]
        log_conversation(assistant_conversation)

        logger.info(
            f"Generating assistant response {episode.step_count + 1} with {self.generation_model_names['generation']}"
        )
        assistant_response_result = await generation_wrapper.generate(
            [assistant_conversation]
        )
        assistant_response = assistant_response_result[0]

        episode.conversation.append(
            {"role": "assistant", "content": assistant_response}
        )

        # Store the latest responses
        if episode.user_responses is None:
            episode.user_responses = []
        episode.user_responses.append(user_response)

        logger.info(
            f"Generated turn {episode.step_count + 1}: User={user_response[:50]}..., Assistant={assistant_response[:50]}..."
        )

    def get_output_row(self, episode: RPGEpisode) -> list[dict]:
        scenario = episode.scenario or "No scenario generated"
        return [
            {
                "game_setting": episode.game_setting or "No setting generated",
                "player_character": episode.player_character
                or "No characters generated",
                "scenario": scenario,
                "user_responses": episode.user_responses[0]
                if episode.user_responses and len(episode.user_responses) > 0
                else "No user responses generated",
                "original_input": {"prompt": self.input_prompt},
                "generation_model": self.generation_model_names["generation"],
                "followup_model": self.generation_model_names["followup"],
                "parameter_model": self.generation_model_names["parameter"],
            }
        ]
