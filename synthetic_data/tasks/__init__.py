from abc import ABC
from typing import Dict, List, Optional
from datasets import Dataset
from rich.console import Console
from evaluation.code_execution import (
    EvalTask,
)
from synthetic_data.generation import GenWrapperArgs
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

    eval_tasks: List[EvalTask] = []

    gen_wrapper_args_override: Optional[GenWrapperArgs] = None

    def load_custom(self) -> Dataset:
        """
        Custom dataset loading logic. Only used if seed_data_format is DatasetFormat.CUSTOM.
        """
        raise NotImplementedError

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Any preprocessing that needs to be done on the dataset before it is used.
        """
        return dataset

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def format_output_rows(self, completions: List[str]) -> List:
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
