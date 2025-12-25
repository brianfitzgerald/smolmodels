import os
from abc import ABC
from typing import Generic, List, Literal, Optional, TypeVar

import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    GenWrapperArgs,
    RemoteModel,
    get_generation_wrapper,
)
from synthetic_data.generation_utils import GenerationRole
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
)

RunMode = Literal["modal", "cli", "notebook"]

# Type for the input to the initial_observation function
SampleT = TypeVar("SampleT")
# Type for the finished episode
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
            results = await generation_wrapper.generate(conversations)
            # Extract content from GenerationResult objects
            completions = [r.content or "" for r in results]
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
    """
    Base task class for multi-step tasks.
    EpisodeT is the type for the finished episode
    SampleT is the type for the input sample data.
    """

    seed_data_location: str | None = None
    seed_data_format: DatasetFormat = DatasetFormat.SYNTHETIC
    output_dataset_name: str
    output_dataset_org: str = "roborovski"
    output_dataset_format: DatasetFormat = DatasetFormat.HF_DATASET

    dataset_columns: List[str] = []

    run_mode: RunMode

    generation_wrappers: dict[GenerationRole, GenerationWrapper] = {}
    generation_model_names: dict[GenerationRole, RemoteModel] = {}

    def __init__(self, run_mode: RunMode) -> None:
        self.run_mode = run_mode
        self.dataset_root_path = (
            "../dataset_files"
            if run_mode == "notebook"
            else "./dataset_files"
            if run_mode == "cli"
            else "/dataset_files"
        )

    def _add_generation_wrapper(self, role: GenerationRole, model: RemoteModel):
        self.generation_wrappers[role] = get_generation_wrapper(model)
        self.generation_model_names[role] = model

    def load_dataset(self) -> Dataset:
        """
        Load the seed dataset based on the specified format.
        """
        logger.info(
            f"Loading dataset with seed_data_location={self.seed_data_location}, seed_data_format={self.seed_data_format}"
        )
        if self.seed_data_format == DatasetFormat.CUSTOM:
            logger.info("Using CUSTOM format, calling load_custom")
            return self.load_custom(self.dataset_root_path)
        elif self.seed_data_location is None:
            logger.info("seed_data_location is None, returning empty dataset")
            return Dataset.from_dict({k: [] for k in self.dataset_columns})
        else:
            assert self.seed_data_location, (
                f"Input dataset location must be provided, but is {self.seed_data_location}"
            )
            if self.seed_data_format == DatasetFormat.HF_DATASET:
                return load_dataset(self.seed_data_location, split="train")  # type: ignore
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

    def load_custom(self, dataset_root_path: str) -> Dataset:
        """
        Custom dataset loading logic. Only used if seed_data_format is DatasetFormat.CUSTOM.
        """
        raise NotImplementedError

    async def initial_step(self, sample: SampleT) -> EpisodeT:
        """
        Initial step of the task. Perform setup; cannot exit from here.
        """
        raise NotImplementedError

    async def step(self, episode: EpisodeT) -> tuple[EpisodeT, bool]:
        """
        Step the episode. Returns the episode and a boolean indicating if the episode is finished.
        """
        raise NotImplementedError

    def format_episode(self, episode: EpisodeT) -> dict:
        """
        Convert a finished episode to a dictionary for storage in a dataset.
        """
        raise NotImplementedError
