import asyncio
from typing import Dict, List, Optional, cast

from dotenv import load_dotenv
import fire
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi
from loguru import logger
from tqdm import tqdm
import os

from gyms.twenty_questions.env import TextEnv
from synthetic_data.generation import (
    get_generation_wrapper,
    MockGenerator,
    RemoteModel,
    save_output_dataset,
)
from synthetic_data.tasks import BaseTask
from synthetic_data.tasks.writing import (
    GutenbergExtraction,
    GutenbergBacktranslation,
    ScreenplaySummarize,
    WritingRewardAnnotate,
)
from synthetic_data.utils import DatasetFormat, print_result_dicts
from gyms import TwentyQuestionsPolicyEnvironment


ALL_TASKS: Dict[str, type[BaseTask]] = {
    "screenplay_summarize": ScreenplaySummarize,
    "gutenberg_extraction": GutenbergExtraction,
    "gutenberg_backtranslation": GutenbergBacktranslation,
    "writing_reward": WritingRewardAnnotate,
}


ALL_ENVIRONMENTS: dict[str, type[TextEnv]] = {
    "twenty_questions": TwentyQuestionsPolicyEnvironment,
}


def main(
    task_name: str,
    environment_name: str,
    save_every_n_batches: int = 5,
    batch_size: int = 1,
    restart: bool = False,
    resume_input_position: bool = True,
    model: str = RemoteModel.DEEPSEEK_V3.value,
    n_epochs: int = 1,
    dataset_root_path: str = "dataset_files",
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"
    assert task_name is not None, "Task name must be passed"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    task = ALL_TASKS[task_name]()
    split = task.seed_data_split

    load_dotenv(".env")
    generation_wrapper = get_generation_wrapper(model, task.gen_wrapper_args_override)
    output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    if task_name and environment_name:
        raise ValueError("Only one of task_name or environment should be passed")

    if environment_name:
        env = ALL_ENVIRONMENTS[environment_name]()
        env.reset()
        return

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

    input_dataset: Dataset
    input_dataset_location: Optional[str] = None
    if task.seed_data_format in (
        DatasetFormat.HF_DATASET,
        DatasetFormat.PARQUET,
        DatasetFormat.TSV,
    ):
        input_dataset_location = task.seed_data_location

    logger.info(
        f"Loading input dataset: {input_dataset_location}, format: {task.seed_data_format.value}"
    )
    if task.seed_data_format == DatasetFormat.CUSTOM:
        input_dataset = task.load_custom()
    else:
        assert input_dataset_location, (
            f"Input dataset location must be provided, but is {input_dataset_location}"
        )
        if task.seed_data_format == DatasetFormat.HF_DATASET:
            input_dataset = cast(
                Dataset, load_dataset(input_dataset_location, split=split)
            )
        elif task.seed_data_format == DatasetFormat.TSV:
            seed_data = pd.read_csv(input_dataset_location, on_bad_lines="skip")
            input_dataset = Dataset.from_pandas(seed_data)
        elif task.seed_data_format == DatasetFormat.PARQUET:
            input_dataset = Dataset.from_parquet(input_dataset_location)  # type: ignore
        else:
            raise ValueError(f"Unrecognized seed_data_format: {task.seed_data_format}")

    # Resume from position

    if resume_input_position and len(output_dataset) > 0:
        # skip first n rows
        logger.info(f"Resuming from position {len(output_dataset)}")
        input_dataset = input_dataset.skip(len(output_dataset))

    input_dataset = task.preprocess_dataset(input_dataset)

    logger.info(
        f"Input dataset length: {len(input_dataset)} Output dataset: {len(output_dataset)}"
    )
    all_new_dataset_rows: List[Dict] = []
    logger.info(f"Generating with model {model} for task {task_name}")

    n_batches = len(input_dataset) // batch_size

    # Generation loop

    for _ in range(n_epochs):
        for batch_idx, batch in enumerate(
            tqdm(input_dataset.iter(batch_size=batch_size), total=n_batches)
        ):
            batch = cast(Dict, batch)
            conversations_batch = task.format_input_conversation(batch)
            if len(conversations_batch) == 0:
                logger.warning(f"Skipping empty batch {batch_idx}")
                continue

            if isinstance(generation_wrapper, MockGenerator):
                generation_wrapper.set_mock_completions(
                    [
                        "def solution(problem_input):\n    return []"
                        for _ in range(len(conversations_batch))
                    ]
                )

            logger.info(
                f"Generating batch of {len(conversations_batch)} completions..."
            )
            try:
                completions = asyncio.run(
                    generation_wrapper.generate(conversations_batch)
                )
            except TimeoutError:
                logger.error(f"Timeout error on batch {batch_idx}")
                continue

            output_rows_batch = task.format_output_rows(completions)
            print_result_dicts(output_rows_batch)
            all_new_dataset_rows.extend(output_rows_batch)
            if batch_idx % save_every_n_batches == 0 and batch_idx > 0:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    all_new_dataset_rows,
                    task.output_dataset_format,
                    dataset_root_path,
                )


if __name__ == "__main__":
    fire.Fire(main)
