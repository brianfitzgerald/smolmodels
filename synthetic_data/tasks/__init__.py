from abc import ABC
from typing import List, Optional, TypeVar, Generic
from loguru import logger
from synthetic_data.generation import GenWrapperArgs, GenerationWrapper
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
)
from datasets import Dataset, load_dataset
import pandas as pd
import os
from typing import Literal

RunMode = Literal["modal", "cli", "notebook"]

# Generic type for sample dataclasses
SampleT = TypeVar("SampleT")
EpisodeT = TypeVar("EpisodeT")


class BaseTaskV1(ABC):
    seed_data_format: DatasetFormat = DatasetFormat.SYNTHETIC
    seed_data_split = "train"

    seed_data_location: str | None = None
    output_dataset_name: str
    output_dataset_org: str = "roborovski"
    output_dataset_format: DatasetFormat = DatasetFormat.HF_DATASET

    dataset_columns: List[str] = []

    gen_wrapper_args_override: Optional[GenWrapperArgs] = None

    run_mode: RunMode

    def __init__(self, run_mode: RunMode) -> None:
        self.run_mode = run_mode
        self.dataset_root_path = (
            "../dataset_files"
            if run_mode == "notebook"
            else "./dataset_files"
            if run_mode == "cli"
            else "/dataset_files"
        )

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

    def load_dataset(self) -> Dataset:
        """
        Load the seed dataset based on the specified format.
        """
        if self.seed_data_location is None:
            return Dataset.from_dict({k: [] for k in self.dataset_columns})
        elif self.seed_data_format == DatasetFormat.CUSTOM:
            return self.load_custom(self.dataset_root_path)
        else:
            assert self.seed_data_location, (
                f"Input dataset location must be provided, but is {self.seed_data_location}"
            )
            if self.seed_data_format == DatasetFormat.HF_DATASET:
                return load_dataset(self.seed_data_location, split=self.seed_data_split)  # type: ignore
            elif self.seed_data_format == DatasetFormat.TSV:
                seed_data = pd.read_csv(
                    os.path.join(
                        self.dataset_root_path, f"{self.seed_data_location}.tsv"
                    ),
                    on_bad_lines="skip",
                )
                return Dataset.from_pandas(seed_data)
            elif self.seed_data_format == DatasetFormat.PARQUET:
                return Dataset.from_parquet(
                    os.path.join(
                        self.dataset_root_path, f"{self.seed_data_location}.parquet"
                    )
                )  # type: ignore
            else:
                raise ValueError(
                    f"Unrecognized seed_data_format: {self.seed_data_format}"
                )


class BaseTask(ABC, Generic[SampleT, EpisodeT]):
    seed_data_location: str | None = None
    output_dataset_name: str
    output_dataset_org: str = "roborovski"
    output_dataset_format: DatasetFormat = DatasetFormat.HF_DATASET

    dataset_columns: List[str] = []

    run_mode: RunMode

    def __init__(self, run_mode: RunMode) -> None:
        self.run_mode = run_mode
        self.dataset_root_path = (
            "../dataset_files"
            if run_mode == "notebook"
            else "./dataset_files"
            if run_mode == "cli"
            else "/dataset_files"
        )

    async def start_episode(self, sample: SampleT) -> EpisodeT:
        raise NotImplementedError

    async def step_episode(self, episode: EpisodeT) -> list[dict]:
        raise NotImplementedError

    def get_output_row(self, episode: EpisodeT) -> list[dict]:
        raise NotImplementedError
