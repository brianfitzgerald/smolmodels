from abc import ABC
from typing import Dict, List, Optional
from datasets import Dataset
from loguru import logger
from rich.console import Console
from evaluation.code_execution import (
    EvalTask,
)
from synthetic_data.generation import GenWrapperArgs, GenerationWrapper
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
)


class BaseTask(ABC):
    seed_data_format: DatasetFormat = DatasetFormat.SYNTHETIC
    seed_data_split = "train"

    seed_data_location: str
    output_dataset_name: str
    output_dataset_org: str = "roborovski"
    output_dataset_format: DatasetFormat = DatasetFormat.HF_DATASET

    dataset_columns: List[str] = []

    gen_wrapper_args_override: Optional[GenWrapperArgs] = None

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic. Only used if seed_data_format is DatasetFormat.CUSTOM.
        """
        raise NotImplementedError

    async def preprocess_row(self, row: dict) -> list[dict]:
        """
        Preprocess a row of data from the dataset.
        """
        return [row]

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        return []

    def format_output_rows(
        self, completions: list[str], input_rows: list[dict]
    ) -> list[dict]:
        """
        Take the completed conversation and format it into the final dataset format.
        """
        raise NotImplementedError

    def format_inference_conversation(
        self, sample: Dict, eval_task: Optional[EvalTask] = None
    ) -> Conversation:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    async def generate(
        self, generation_wrapper: GenerationWrapper, input_rows: list[dict]
    ) -> list[dict]:
        try:
            conversations = self.format_input_conversation(input_rows)
            completions = await generation_wrapper.generate(conversations)
            return self.format_output_rows(completions, input_rows)
        except TimeoutError:
            logger.error("Timeout error processing batch")
            return []
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []
