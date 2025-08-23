from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
    get_generation_wrapper,
    GenWrapperArgs,
)
from synthetic_data.tasks import BaseTask, RunMode

from synthetic_data.utils import DatasetFormat, Conversation


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


class RoleplayingGameMultiStepTask(BaseTask):
    """
    Multi-step variant: Step 1 generates a scenario; Step 2 generates follow-up questions.
    """

    def __init__(
        self,
        run_mode: RunMode,
        generation_model: RemoteModel = "gpt-4o-mini",
        followup_model: RemoteModel = "gpt-4.1-nano",
        num_followups: int = 3,
        input_prompt: str = "A mysterious forest adventure",
    ):
        super().__init__(run_mode)
        self.generation_model = generation_model
        self.followup_model = followup_model
        self.num_followups = num_followups
        self.input_prompt = input_prompt

        # adopt dataset config from RoleplayingGame single-step task
        self.task = RoleplayingGame(run_mode)
        self.output_dataset_name = self.task.output_dataset_name
        self.output_dataset_org = self.task.output_dataset_org
        self.output_dataset_format = self.task.output_dataset_format
        self.dataset_columns = self.task.dataset_columns

    class _Episode:
        step_count: int
        scenario: str | None
        followups: str | None
        conversation: Conversation
        run_metadata: dict
        seed: int
        followup_wrapper: GenerationWrapper

        def __init__(self, seed: int, followup_wrapper: GenerationWrapper):
            self.step_count = 0
            self.scenario = None
            self.followups = None
            self.conversation = []
            self.run_metadata = {}
            self.seed = seed
            self.followup_wrapper = followup_wrapper

    def new_episode(self, generation_wrapper: GenerationWrapper, seed: int):
        followup_wrapper = get_generation_wrapper(
            self.followup_model, args_override=GenWrapperArgs(seed=seed)
        )
        ep = RoleplayingGameMultiStepTask._Episode(seed, followup_wrapper)
        ep.run_metadata = {
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
            "num_followups": self.num_followups,
            "input_prompt": self.input_prompt,
            "seed": seed,
            "scenario_generated": False,
            "followups_generated": False,
        }
        logger.info(f"Start episode with prompt: {self.input_prompt}, seed: {seed}")
        return ep

    async def step_episode(
        self,
        generation_wrapper: GenerationWrapper,
        ep: "RoleplayingGameMultiStepTask._Episode",
    ) -> bool:
        if ep.step_count == 0:
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
            scenario_response = await generation_wrapper.generate(
                [scenario_conversation]
            )
            ep.scenario = scenario_response[0]
            ep.conversation.extend(scenario_conversation)
            ep.conversation.append({"role": "assistant", "content": ep.scenario})
            ep.run_metadata["scenario_generated"] = True
            ep.step_count += 1
            logger.info(f"Generated scenario: {ep.scenario[:100]}...")
            return False

        elif ep.step_count == 1:
            # Step 2: Generate follow-up questions
            followup_conversation: Conversation = [
                {
                    "role": "system",
                    "content": f"You are a game master creating follow-up questions for a roleplaying scenario. Generate exactly {self.num_followups} thought-provoking questions that players could explore in this scenario. Format as a numbered list.",
                },
                {
                    "role": "user",
                    "content": f"Scenario: {ep.scenario}\n\nGenerate {self.num_followups} follow-up questions for this scenario.",
                },
            ]

            logger.info(f"Generating follow-up questions with {self.followup_model}")
            followup_response = await ep.followup_wrapper.generate(
                [followup_conversation]
            )
            ep.followups = followup_response[0]
            ep.conversation.extend(followup_conversation)
            ep.conversation.append({"role": "assistant", "content": ep.followups})
            ep.run_metadata["followups_generated"] = True
            ep.step_count += 1
            logger.info(f"Generated follow-ups: {ep.followups[:100]}...")
            return True

        return True

    def get_output_row(self, ep: "RoleplayingGameMultiStepTask._Episode") -> dict:
        if ep.scenario is None:
            return {
                "scenario": "No scenario generated",
                "follow_ups": "No follow-ups generated",
                "original_input": {"prompt": self.input_prompt},
                "generation_model": self.generation_model,
                "followup_model": self.followup_model,
            }

        return {
            "scenario": ep.scenario,
            "follow_ups": ep.followups
            if ep.followups is not None
            else "No follow-ups generated",
            "original_input": {"prompt": self.input_prompt},
            "generation_model": self.generation_model,
            "followup_model": self.followup_model,
        }

    # No explicit flag; episodes are discovered by runner
