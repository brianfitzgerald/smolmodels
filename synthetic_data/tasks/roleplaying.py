from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
    get_generation_wrapper,
    GenWrapperArgs,
)
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import DatasetFormat, Conversation
from gyms.utils import TextEnv


class RoleplayingGame(BaseTask):
    """
    Generate roleplaying scenarios with follow-up questions using different models.
    """

    output_dataset_name = "roleplaying_scenarios"
    dataset_columns = [
        "scenario",
        "follow_ups",
        "original_input",
        "generation_model",
        "followup_model",
    ]
    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_format = DatasetFormat.PARQUET


class RoleplayingGameEnvironment(TextEnv):
    """
    TextEnv for generating roleplaying scenarios with follow-up questions.
    On each step:
    1. Generate a roleplaying scenario based on the input prompt
    2. Generate follow-up questions for the scenario
    """

    def __init__(
        self,
        generation_wrapper: GenerationWrapper,
        seed: int,
        generation_model: RemoteModel = "gpt-4o-mini",
        followup_model: RemoteModel = "gpt-4.1-nano",
        num_followups: int = 3,
        input_prompt: str = "A mysterious forest adventure",
    ):
        super().__init__(generation_wrapper, seed)
        self.generation_model = generation_model
        self.followup_model = followup_model
        self.num_followups = num_followups
        self.input_prompt = input_prompt

        # Initialize the followup model wrapper
        self.followup_wrapper = get_generation_wrapper(
            self.followup_model,
            args_override=GenWrapperArgs(seed=self.seed),
        )

        # Initialize state
        self.step_count = 0
        self.scenario = None
        self.followups = None
        self.conversation: Conversation = []
        self.task = RoleplayingGame(run_mode="modal")
        self.run_metadata: dict = {}

        # Track completion state
        self.scenario_generated = False
        self.followups_generated = False

    async def step(self) -> bool:
        """
        Perform a single step in the environment.
        Step 1: Generate scenario
        Step 2: Generate follow-ups
        Returns True when episode is done.
        """
        if self.step_count == 0:
            # Step 1: Generate the roleplaying scenario
            scenario_conversation: Conversation = [
                {
                    "role": "system",
                    "content": "You are a creative storyteller. Generate an engaging roleplaying scenario based on the given prompt. Be descriptive and immersive.",
                },
                {
                    "role": "user",
                    "content": f"Create a roleplaying scenario based on this: {self.input_prompt}",
                },
            ]

            logger.info(f"Generating scenario with {self.generation_model}")
            scenario_response = await self.generation_wrapper.generate(
                [scenario_conversation]
            )
            self.scenario = scenario_response[0]
            self.conversation.extend(scenario_conversation)
            self.conversation.append({"role": "assistant", "content": self.scenario})
            self.scenario_generated = True
            self.step_count += 1

            logger.info(f"Generated scenario: {self.scenario[:100]}...")
            return False

        elif self.step_count == 1:
            # Step 2: Generate follow-up questions
            followup_conversation: Conversation = [
                {
                    "role": "system",
                    "content": f"You are a game master creating follow-up questions for a roleplaying scenario. Generate exactly {self.num_followups} thought-provoking questions that players could explore in this scenario. Format as a numbered list.",
                },
                {
                    "role": "user",
                    "content": f"Scenario: {self.scenario}\n\nGenerate {self.num_followups} follow-up questions for this scenario.",
                },
            ]

            logger.info(f"Generating follow-up questions with {self.followup_model}")
            followup_response = await self.followup_wrapper.generate(
                [followup_conversation]
            )
            self.followups = followup_response[0]
            self.conversation.extend(followup_conversation)
            self.conversation.append({"role": "assistant", "content": self.followups})
            self.followups_generated = True
            self.step_count += 1

            logger.info(f"Generated follow-ups: {self.followups[:100]}...")
            return True  # Episode is complete

        return True  # Should not reach here

    def reset(self):
        """Reset the environment to the initial state."""
        self.step_count = 0
        self.scenario = None
        self.followups = None
        self.conversation = []
        self.scenario_generated = False
        self.followups_generated = False

        # Update metadata
        self.run_metadata = {
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
            "num_followups": self.num_followups,
            "input_prompt": self.input_prompt,
            "seed": self.seed,
            "scenario_generated": False,
            "followups_generated": False,
        }

        logger.info(
            f"Reset environment with prompt: {self.input_prompt}, seed: {self.seed}"
        )

    def get_output_data(self) -> dict:
        """Get the generated data in the format expected by the task."""
        if not self.scenario_generated:
            return {
                "scenario": "No scenario generated",
                "follow_ups": "No follow-ups generated",
                "original_input": {"prompt": self.input_prompt},
                "generation_model": self.generation_model,
                "followup_model": self.followup_model,
            }

        return {
            "scenario": self.scenario,
            "follow_ups": self.followups
            if self.followups_generated
            else "No follow-ups generated",
            "original_input": {"prompt": self.input_prompt},
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
        }
