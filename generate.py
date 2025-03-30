import asyncio
from collections.abc import AsyncGenerator
from copy import copy
from typing import Dict, Optional, cast
from huggingface_hub import login
from typing import Literal

from dotenv import load_dotenv
import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi
from loguru import logger
import os

from gyms.run import run_environments
from gyms.twenty_questions.env import TextEnv
from synthetic_data.generation import (
    GenWrapperArgs,
    get_generation_wrapper,
    RemoteModel,
    save_output_dataset,
)
from synthetic_data.tasks import BaseTask
from synthetic_data.tasks.writing import (
    GutenbergBacktranslationFromTxt,
    GutenbergExtraction,
    GutenbergBacktranslation,
    ScreenplaySummarize,
    BacktranslateBestOfN,
)
from synthetic_data.utils import DatasetFormat, print_result_dicts
from gyms import TwentyQuestionsPolicyEnvironment


TaskName = Literal[
    "screenplay_summarize",
    "gutenberg_extraction",
    "gutenberg_backtranslation",
    "gutenberg_backtranslation_from_txt",
    "backtranslate_best_of_n",
]
ALL_TASKS: Dict[TaskName, type[BaseTask]] = {
    "screenplay_summarize": ScreenplaySummarize,
    "gutenberg_extraction": GutenbergExtraction,
    "gutenberg_backtranslation": GutenbergBacktranslation,
    "gutenberg_backtranslation_from_txt": GutenbergBacktranslationFromTxt,
    "backtranslate_best_of_n": BacktranslateBestOfN,
}


ALL_ENVIRONMENTS: dict[str, type[TextEnv]] = {
    "twenty_questions": TwentyQuestionsPolicyEnvironment,
}


async def process_batch(
    task: BaseTask, generation_wrapper, input_rows: list[dict]
) -> list[dict]:
    logger.info(f"Processing batch of {len(input_rows)} rows")
    output_rows = await task.generate(generation_wrapper, input_rows)
    if len(output_rows) == 0:
        logger.warning("Skipping empty batch")
        return []
    return output_rows


async def collect_preprocessed_rows(
    task: BaseTask, input_dataset: Dataset, batch_size: int, max_input_buffer: int = 10
) -> AsyncGenerator[list[dict], None]:
    input_buffer = []
    current_position = 0
    results = []
    while current_position < len(input_dataset):
        # Fill input buffer
        while len(input_buffer) < max_input_buffer and current_position < len(
            input_dataset
        ):
            input_buffer.append(input_dataset[current_position])
            current_position += 1

        if not input_buffer:
            break

        logger.info(f"Processing buffer of {len(input_buffer)} rows")
        # Process current buffer
        results = []
        for row in input_buffer:
            result = await task.preprocess_row(row)
            if result:
                results.extend(result)
        input_buffer.clear()

        # Yield batches of preprocessed rows
        while len(results) >= batch_size:
            yield results[:batch_size]
            results = results[batch_size:]

    # Yield any remaining rows
    if results:
        yield results


async def process_dataset(
    task: BaseTask,
    generation_wrapper,
    input_dataset: Dataset,
    batch_size: int,
    save_every_n_batches: int,
    output_dataset: Dataset,
    dataset_root_path: str,
) -> list[dict]:
    all_new_dataset_rows = []
    batch_count = 0

    preprocessed_batches = collect_preprocessed_rows(task, input_dataset, batch_size)
    async for preprocessed_batch in preprocessed_batches:
        # Process whatever number of rows we got, even if less than batch_size
        output_rows = await process_batch(task, generation_wrapper, preprocessed_batch)
        if output_rows:
            print_result_dicts(output_rows)
            all_new_dataset_rows.extend(output_rows)

        batch_count += 1
        if batch_count % save_every_n_batches == 0:
            save_output_dataset(
                output_dataset,
                task.output_dataset_name,
                all_new_dataset_rows,
                task.output_dataset_format,
                dataset_root_path,
            )

    return all_new_dataset_rows


def main(
    task_name: TaskName | None = None,
    environment_name: str | None = None,
    save_every_n_batches: int = 5,
    batch_size: int = 8,
    restart: bool = False,
    resume_input_position: bool = True,
    model: RemoteModel = "gemini-2.0-flash",
    n_epochs: int = 1,
    dataset_root_path: str = "dataset_files",
    running_on_modal: bool = False,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"
    assert not (task_name and environment_name), (
        "Only one of task_name or environment_name should be passed"
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    load_dotenv(".env")
    args_override: GenWrapperArgs | None = None
    generation_wrapper = get_generation_wrapper(model, args_override)
    environment: Optional[TextEnv] = None

    if "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token, add_to_git_credential=True)

    if task_name:
        task = ALL_TASKS[task_name]()
        task.run_mode = "cli" if not running_on_modal else "modal"
    else:
        assert environment_name, "Environment name must be passed"
        environment = ALL_ENVIRONMENTS[environment_name](generation_wrapper, 0)
        task = environment.task
        args_override = task.gen_wrapper_args_override

    output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    if task_name and environment_name:
        raise ValueError("Only one of task_name or environment should be passed")
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
    logger.info(
        f"Loading input dataset: {task.seed_data_location}, format: {task.seed_data_format.value}"
    )
    input_dataset: Dataset = task.load_dataset()

    # Resume from position

    if resume_input_position and len(output_dataset) > 0:
        # skip first n rows
        logger.info(f"Resuming from position {len(output_dataset)}")
        input_dataset = input_dataset.skip(len(output_dataset))

    logger.info(
        f"Input dataset length: {len(input_dataset)} Output dataset: {len(output_dataset)}"
    )

    # Generation loop for generation tasks
    if not environment_name:
        for _ in range(n_epochs):
            all_new_dataset_rows = asyncio.run(
                process_dataset(
                    task,
                    generation_wrapper,
                    input_dataset,
                    batch_size,
                    save_every_n_batches,
                    output_dataset,
                    dataset_root_path,
                )
            )

            # Final save of any remaining rows
            if all_new_dataset_rows:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    all_new_dataset_rows,
                    task.output_dataset_format,
                    dataset_root_path,
                )
    else:
        assert environment, "Environment must be passed"
        envs = [copy(environment) for _ in range(batch_size)]
        asyncio.run(
            run_environments(
                envs,
                n_epochs,
                save_every_n_batches,
                dataset_root_path=dataset_root_path,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
