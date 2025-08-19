from typing import List

from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
    get_generation_wrapper,
)
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import DatasetFormat, Conversation
from synthetic_data.tasks import RunMode


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

    def __init__(
        self,
        run_mode: RunMode,
        generation_model: RemoteModel = "gpt-4o-mini",
        followup_model: RemoteModel = "gpt-4.1-nano",
        num_followups: int = 3,
    ) -> None:
        super().__init__(run_mode)
        self.num_followups = num_followups
        self.generation_model = generation_model
        self.followup_model = followup_model


    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        """Format input data into conversations for the generation model."""
        conversations = []
        for row in batch:
            # Assume input has a 'prompt' or 'input' field
            prompt = row.get("prompt", row.get("input", str(row)))
            conversation = [
                {
                    "role": "system",
                    "content": "You are a creative storyteller. Generate an engaging roleplaying scenario based on the given prompt. Be descriptive and immersive.",
                },
                {
                    "role": "user",
                    "content": f"Create a roleplaying scenario based on this: {prompt}",
                },
            ]
            conversations.append(conversation)
        return conversations

    def _format_followup_conversations(
        self, scenarios: List[str], original_inputs: List[dict]
    ) -> List[Conversation]:
        """Format scenarios into conversations for follow-up question generation."""
        conversations = []
        for scenario, _ in zip(
            scenarios, original_inputs
        ):  # _ instead of original_input to avoid unused warning
            conversation = [
                {
                    "role": "system",
                    "content": f"You are a game master creating follow-up questions for a roleplaying scenario. Generate exactly {self.num_followups} thought-provoking questions that players could explore in this scenario. Format as a numbered list.",
                },
                {
                    "role": "user",
                    "content": f"Scenario: {scenario}\n\nGenerate {self.num_followups} follow-up questions for this scenario.",
                },
            ]
            conversations.append(conversation)
        return conversations

    def format_output_rows(
        self, completions: List[str], input_rows: List[dict]
    ) -> List[dict]:
        """Format completions into output rows - compatibility with BaseTask interface."""
        # For compatibility with BaseTask, treat completions as scenarios
        # This method is called when only initial generation is done
        output_rows = []
        for scenario, input_row in zip(completions, input_rows):
            output_rows.append(
                {
                    "scenario": scenario,
                    "follow_ups": "No follow-ups generated yet",
                    "original_input": input_row,
                    "generation_model": self.generation_model,
                    "followup_model": self.followup_model,
                }
            )
        return output_rows

    def format_complete_output_rows(
        self, scenarios: List[str], followups: List[str], input_rows: List[dict]
    ) -> List[dict]:
        """Format the generated scenarios and follow-ups into output rows."""
        output_rows = []
        for scenario, followup, input_row in zip(scenarios, followups, input_rows):
            output_rows.append(
                {
                    "scenario": scenario,
                    "follow_ups": followup,
                    "original_input": input_row,
                    "generation_model": self.generation_model,
                    "followup_model": self.followup_model,
                }
            )
        return output_rows

    async def generate(
        self, generation_wrapper: GenerationWrapper, input_rows: list[dict]
    ) -> list[dict]:
        """Generate roleplaying scenarios and follow-up questions using potentially different models."""
        try:
            # Step 1: Generate initial scenarios
            conversations = self.format_input_conversation(input_rows)

            logger.info(
                f"Generating {len(conversations)} scenarios with {self.generation_model}"
            )
            
            # Use the passed generation wrapper for scenarios
            scenarios = await generation_wrapper.generate(conversations)

            if not scenarios or len(scenarios) != len(input_rows):
                logger.error(
                    f"Generation failed: expected {len(input_rows)} scenarios, got {len(scenarios)}"
                )
                return []

            # Step 2: Generate follow-up questions using the followup model
            followup_wrapper = get_generation_wrapper(self.followup_model)
            followup_conversations = self._format_followup_conversations(
                scenarios, input_rows
            )

            logger.info(
                f"Generating follow-up questions with {self.followup_model}"
            )
            followups = await followup_wrapper.generate(followup_conversations)

            if not followups or len(followups) != len(scenarios):
                logger.error(
                    f"Follow-up generation failed: expected {len(scenarios)} follow-ups, got {len(followups)}"
                )
                # Fall back to empty follow-ups rather than failing completely
                followups = ["No follow-up questions generated"] * len(scenarios)

            return self.format_complete_output_rows(scenarios, followups, input_rows)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []

