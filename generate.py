import asyncio
import os
from typing import Dict, Literal, cast

import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger

from synthetic_data.generation import (
    GenWrapperArgs,
    RemoteModel,
    get_generation_wrapper,
)
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.multistep import run_task
from synthetic_data.tasks.roleplaying import (
    RoleplayingGame,
    RoleplayingGameMultiStepTask,
)
from synthetic_data.tasks.writing import (
    GenerationBestOfN,
    GutenbergBacktranslation,
    GutenbergBacktranslationFromTxt,
    GutenbergExtraction,
    ScreenplaySummarize,
)
from synthetic_data.utils import DatasetFormat

TaskName = Literal[
    "screenplay_summarize",
    "gutenberg_extraction",
    "gutenberg_backtranslation",
    "gutenberg_backtranslation_from_txt",
    "generation_best_of_n",
    "roleplaying_game",
    "roleplaying_game_env",
]
ALL_TASKS: Dict[TaskName, type[BaseTask]] = {
    "screenplay_summarize": ScreenplaySummarize,
    "gutenberg_extraction": GutenbergExtraction,
    "gutenberg_backtranslation": GutenbergBacktranslation,
    "gutenberg_backtranslation_from_txt": GutenbergBacktranslationFromTxt,
    "generation_best_of_n": GenerationBestOfN,
    "roleplaying_game": RoleplayingGame,
    "roleplaying_game_env": RoleplayingGameMultiStepTask,
}


def main(
    task_name: TaskName | None = None,
    save_every_n_batches: int = 5,
    batch_size: int = 16,
    restart: bool = False,
    resume_input_position: bool = True,
    model: RemoteModel = "gemini-2.0-flash",
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
    args_override: GenWrapperArgs | None = None
    generation_wrapper = get_generation_wrapper(model, args_override)
    # no environments — tasks can be multi-step

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

    if task.seed_data_location is not None:
        # Load input dataset
        logger.info(
            f"Loading input dataset from location: {task.seed_data_location}, format: {task.seed_data_format.value}"
        )
    input_dataset: Dataset = task.load_dataset()
    logger.info(f"Input dataset length: {len(input_dataset)}")

    # Resume from position

    if resume_input_position and len(output_dataset) > 0:
        # skip first n rows
        logger.info(f"Resuming from position {len(output_dataset)}")
        input_dataset = input_dataset.skip(len(output_dataset))

    logger.info(
        f"Input dataset length: {len(input_dataset)} Output dataset: {len(output_dataset)}"
    )
    out_dir = (
        "../dataset_files"
        if run_mode == "notebook"
        else "./dataset_files"
        if run_mode == "cli"
        else "/model-weights/dataset_files"
    )

    # Determine output directory early (also used by multi-turn path)
    out_dir = (
        "../dataset_files"
        if run_mode == "notebook"
        else "./dataset_files"
        if run_mode == "cli"
        else "/model-weights/dataset_files"
    )

    # Unified run: episodes if implemented, else single-step dataset batching
    asyncio.run(
        run_task(
            task,
            generation_wrapper,
            input_dataset,
            output_dataset,
            batch_size,
            n_epochs,
            save_every_n_batches,
            out_dir,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
