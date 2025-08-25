from jinja2 import Template
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    GenWrapperArgs,
    RemoteModel,
    get_generation_wrapper,
)
from synthetic_data.tasks import BaseTask, BaseTaskV1, RunMode
from synthetic_data.tasks.roleplaying_prompts import (
    GAME_PARAMETER_PROMPT,
    ROLEPLAYING_PROMPT,
)
from synthetic_data.utils import Conversation, DatasetFormat
from datasets import Dataset


class RPGEpisode:
    step_count: int
    game_setting: str | None
    player_characters: str | None
    scenario: str | None
    user_responses: list[str] | None
    conversation: Conversation
    run_metadata: dict
    seed: int

    def __init__(
        self,
        seed: int,
    ):
        self.step_count = 0
        self.game_setting = None
        self.player_characters = None
        self.scenario = None
        self.user_responses = None
        self.conversation = []
        self.run_metadata = {}
        self.seed = seed


class RoleplayingGame(BaseTaskV1):
    """
    Generate roleplaying scenarios with follow-up questions using different models.
    """

    output_dataset_name = "roleplaying_scenarios"
    dataset_columns = [
        "game_setting",
        "player_characters",
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
        generation_model: RemoteModel = "gpt-4o-mini",
        followup_model: RemoteModel = "gpt-4.1-nano",
        parameter_model: RemoteModel = "gpt-4o-mini",
        num_user_responses: int = 3,
        input_prompt: str = "A mysterious forest adventure",
    ):
        super().__init__(run_mode)
        self.generation_model = generation_model
        self.followup_model: RemoteModel = followup_model
        self.parameter_model: RemoteModel = parameter_model
        self.num_user_responses = num_user_responses
        self.input_prompt = input_prompt

        # adopt dataset config from RoleplayingGame single-step task
        self.task = RoleplayingGame(run_mode)
        self.output_dataset_name = self.task.output_dataset_name
        self.output_dataset_org = self.task.output_dataset_org
        self.output_dataset_format = self.task.output_dataset_format
        self.dataset_columns = self.task.dataset_columns
        self.followup_wrapper = get_generation_wrapper(self.followup_model)
        self.parameter_wrapper = get_generation_wrapper(self.parameter_model)
        self.generation_wrapper = get_generation_wrapper(self.generation_model)

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

    async def new_episode(self, sample: None) -> RPGEpisode:
        seed = hash(str(sample)) % (2**32)
        self.followup_wrapper = get_generation_wrapper(
            self.followup_model,
            args_override=GenWrapperArgs(seed=seed),
        )
        self.parameter_wrapper = get_generation_wrapper(
            self.parameter_model,
            args_override=GenWrapperArgs(seed=seed),
        )
        ep = RPGEpisode(seed)
        ep.run_metadata = {
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
            "parameter_model": self.parameter_model,
            "num_user_responses": self.num_user_responses,
            "input_prompt": self.input_prompt,
            "seed": seed,
            "parameters_generated": False,
            "scenario_generated": False,
            "user_responses_generated": False,
        }
        logger.info(f"Start episode with prompt: {self.input_prompt}, seed: {seed}")
        return ep

    async def start_episode(self, sample: None) -> RPGEpisode:
        # Create a new episode with a deterministic seed
        seed = hash(str(sample)) % (2**32)
        # We need a generation_wrapper for the main model, but we don't have it here
        # This will be set later when the episode is used
        ep = RPGEpisode(seed)
        ep.run_metadata = {
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
            "parameter_model": self.parameter_model,
            "num_user_responses": self.num_user_responses,
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
        if not episode.run_metadata.get("parameters_generated", False):
            await self._generate_parameters(episode)
            episode.run_metadata["parameters_generated"] = True
            return []

        if not episode.run_metadata.get("scenario_generated", False):
            await self._generate_scenario(episode, self.generation_wrapper)
            episode.run_metadata["scenario_generated"] = True
            return []

        # Step 1+: Generate turn-by-turn conversation
        if episode.step_count < self.num_user_responses:
            await self._generate_turn(episode, self.followup_wrapper)
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
                "content": f"Generate game parameters for a roleplaying scenario based on this theme: {self.input_prompt}\n\nPlease provide:\n1. A detailed GAME_SETTING (2-3 paragraphs)\n2. PLAYER_CHARACTERS information (1-2 paragraphs describing the main character)",
            },
        ]

        logger.info(f"Generating game parameters with {self.parameter_model}")
        parameter_response = await self.parameter_wrapper.generate(
            [parameter_conversation]
        )

        # Parse the response to extract game_setting and player_characters
        response_text = parameter_response[0]
        # Simple parsing - look for sections
        if (
            "<game_setting>" in response_text.lower()
            and "</game_setting>" in response_text.lower()
        ):
            start = response_text.lower().find("<game_setting>") + len("<game_setting>")
            end = response_text.lower().find("</game_setting>")
            episode.game_setting = response_text[start:end].strip()
        else:
            # Fallback: assume first half is setting, second half is characters
            lines = response_text.split("\n")
            mid_point = len(lines) // 2
            episode.game_setting = "\n".join(lines[:mid_point]).strip()
            episode.player_characters = "\n".join(lines[mid_point:]).strip()

        if (
            "<player_character>" in response_text.lower()
            and "</player_character>" in response_text.lower()
        ):
            start = response_text.lower().find("<player_character>") + len(
                "<player_character>"
            )
            end = response_text.lower().find("</player_character>")
            episode.player_characters = response_text[start:end].strip()

        episode.conversation.extend(parameter_conversation)
        episode.conversation.append({"role": "assistant", "content": response_text})
        logger.info(
            f"Generated game parameters: Setting={episode.game_setting[:50]}..., Characters={episode.player_characters[:50] if episode.player_characters else 'None'}..."
        )

    async def _generate_scenario(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
        """Generate the initial roleplaying scenario"""
        template = Template(ROLEPLAYING_PROMPT)
        formatted_prompt = template.render(
            GAME_SETTING=episode.game_setting or "A mysterious and unknown world",
            PLAYER_CHARACTERS=episode.player_characters
            or "A brave adventurer with unknown abilities",
        )

        scenario_conversation: Conversation = [
            {
                "role": "system",
                "content": formatted_prompt,
            },
            {
                "role": "user",
                "content": "Begin the roleplaying game. Introduce the world and the player's current situation.",
            },
        ]

        logger.info(f"Generating scenario with {self.generation_model}")
        scenario_response = await generation_wrapper.generate([scenario_conversation])
        episode.scenario = scenario_response[0]
        episode.conversation.extend(scenario_conversation)
        episode.conversation.append({"role": "assistant", "content": episode.scenario})
        logger.info(f"Generated scenario: {episode.scenario[:100]}...")

    async def _generate_turn(
        self, episode: RPGEpisode, generation_wrapper: GenerationWrapper
    ):
        """Generate a single turn of conversation (user response + assistant response)"""
        # Generate user response
        user_response_conversation: Conversation = [
            {
                "role": "system",
                "content": "You are simulating a player in a roleplaying game. Based on the scenario and the dungeon master's response, generate a realistic player response that a human player might make. This should be a natural, in-character response that advances the story or explores the scenario.",
            },
            {
                "role": "user",
                "content": f"Game Setting: {episode.game_setting}\nPlayer Character: {episode.player_characters}\nDungeon Master Response: {episode.scenario}\n\nGenerate a realistic player response:",
            },
        ]

        logger.info(
            f"Generating user response {episode.step_count + 1} with {self.followup_model}"
        )
        user_response_result = await self.followup_wrapper.generate(
            [user_response_conversation]
        )
        user_response = user_response_result[0]

        episode.conversation.extend(user_response_conversation)
        episode.conversation.append({"role": "assistant", "content": user_response})

        # Generate assistant response
        assistant_conversation: Conversation = [
            {
                "role": "system",
                "content": "You are a dungeon master in a roleplaying game. Respond to the player's action in character, advancing the story and maintaining the game's atmosphere.",
            },
            {
                "role": "user",
                "content": f"Game Setting: {episode.game_setting}\nPlayer Character: {episode.player_characters}\nPlayer Response: {user_response}\n\nRespond as the dungeon master:",
            },
        ]

        logger.info(
            f"Generating assistant response {episode.step_count + 1} with {self.generation_model}"
        )
        assistant_response_result = await generation_wrapper.generate(
            [assistant_conversation]
        )
        assistant_response = assistant_response_result[0]

        episode.conversation.extend(assistant_conversation)
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
        if episode.scenario is None:
            return [
                {
                    "game_setting": episode.game_setting or "No setting generated",
                    "player_characters": episode.player_characters
                    or "No characters generated",
                    "scenario": "No scenario generated",
                    "user_responses": "No user responses generated",
                    "original_input": {"prompt": self.input_prompt},
                    "generation_model": self.generation_model,
                    "followup_model": self.followup_model,
                    "parameter_model": self.parameter_model,
                }
            ]

        return [
            {
                "game_setting": episode.game_setting or "No setting generated",
                "player_characters": episode.player_characters
                or "No characters generated",
                "scenario": episode.scenario,
                "user_responses": episode.user_responses[0]
                if episode.user_responses and len(episode.user_responses) > 0
                else "No user responses generated",
                "original_input": {"prompt": self.input_prompt},
                "generation_model": self.generation_model,
                "followup_model": self.followup_model,
                "parameter_model": self.parameter_model,
            }
        ]
