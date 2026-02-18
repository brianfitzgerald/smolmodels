import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, cast

import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
)
from synthetic_data.tasks import BaseTask, BaseTaskV1, RunMode
from synthetic_data.tasks.next_chapter import (
    GutenbergSummaryContinuation,
)
from synthetic_data.tasks.roleplaying import (
    RoleplayingGameMultiStepTask,
)
from synthetic_data.utils import DatasetFormat


class PIDController:
    """PID controller for throughput regulation"""

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, setpoint: float, measured_value: float) -> float:
        """Update PID controller and return control output"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Update state
        self.prev_error = error
        self.last_time = current_time

        return p_term + i_term + d_term


@dataclass
class WrapperInfo:
    """Information about a registered wrapper"""

    wrapper: GenerationWrapper
    model_name: str
    provider: str
    max_concurrent: Optional[int] = None


async def run_task(
    task: BaseTask,
    output_dataset: Dataset,
    n_epochs: int,
):
    # TODO implement
    pass


TaskName = Literal[
    "screenplay_summarize",
    "gutenberg_extraction",
    "gutenberg_backtranslation",
    "gutenberg_backtranslation_from_txt",
    "gutenberg_summary_continuation",
    "generation_best_of_n",
    "roleplaying_game",
]
ALL_TASKS: Dict[TaskName, type[BaseTask | BaseTaskV1]] = {
    "roleplaying_game": RoleplayingGameMultiStepTask,
    "gutenberg_summary_continuation": GutenbergSummaryContinuation,
}


def main(
    task_name: TaskName | None = None,
    save_every_n_batches: int = 1,
    batch_size: int = 32,
    restart: bool = False,
    model: RemoteModel = "gpt-5-mini",
    n_epochs: int = 1,
    run_mode: RunMode = "cli",
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    load_dotenv(".env")

    if "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token, add_to_git_credential=True)

    if task_name is None:
        raise ValueError("`task_name` must be provided")

    if task_name in ALL_TASKS:
        task = ALL_TASKS[task_name](run_mode)
    else:
        raise ValueError(f"Unknown task_name: {task_name}")

    output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    # Load output dataset

    logger.info("Loading output dataset...")
    if not restart:
        if task.output_dataset_format == DatasetFormat.HF_DATASET:
            ds_url = f"{task.output_dataset_org}/{task.output_dataset_name}"
            try:
                output_dataset = cast(
                    Dataset,
                    load_dataset(
                        ds_url,
                        split="train",
                    ),
                )
                # TODO filter for rows that don't need completion
            except (EmptyDatasetError, ValueError):
                logger.info("No existing dataset found, starting from scratch...")
            except DatasetNotFoundError:
                logger.info("No existing dataset found, starting from scratch...")
                hf_api = HfApi()
                hf_api.create_repo(
                    repo_id=task.output_dataset_name,
                    repo_type="dataset",
                )
                output_dataset = load_dataset(ds_url, split="train")
        elif task.output_dataset_format == DatasetFormat.PARQUET:
            try:
                output_dataset = cast(
                    Dataset, Dataset.from_parquet(f"{task.output_dataset_name}.parquet")
                )
            except FileNotFoundError:
                logger.info("No existing dataset found, starting from scratch...")
        else:
            output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    output_dataset = cast(Dataset, output_dataset)

    # Load input dataset
    if task.seed_data_location is not None:
        # Load input dataset
        logger.info(
            f"Loading input dataset from location: {task.seed_data_location}, format: {task.seed_data_format.value}"
        )
    input_dataset: Dataset = task.load_dataset()
    logger.info(f"Input dataset length: {len(input_dataset)}")

    out_dir = (
        "../dataset_files"
        if run_mode == "notebook"
        else "./dataset_files"
        if run_mode == "cli"
        else "/model-weights/dataset_files"
    )

    asyncio.run(
        run_task(
            task,  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
            input_dataset,
            output_dataset,
            n_epochs,
        )
    )

    if task.output_dataset_format == DatasetFormat.HF_DATASET:
        output_dataset.push_to_hub(task.output_dataset_name)
    elif task.output_dataset_format == DatasetFormat.PARQUET:
        filename = f"{task.output_dataset_name}.parquet"
        output_dataset.to_parquet(filename)
    else:
        raise ValueError(f"Unsupported output format: {task.output_dataset_format}")


if __name__ == "__main__":
    fire.Fire(main)
