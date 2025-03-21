import asyncio
from copy import copy
import traceback
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
    GenWrapperArgs,
    get_generation_wrapper,
    MockGenerator,
    RemoteModel,
    save_output_dataset,
)
from synthetic_data.tasks import BaseTask
from synthetic_data.tasks.writing import (
    GutenbergBacktranslationFromTxt,
    GutenbergExtraction,
    GutenbergBacktranslation,
    GutenbergFollowUp,
    ScreenplaySummarize,
    WritingRewardAnnotate,
    WritingScoreAnnotate,
)
from synthetic_data.utils import DatasetFormat, dictl, print_result_dicts
from gyms import TwentyQuestionsPolicyEnvironment


ALL_TASKS: Dict[str, type[BaseTask]] = {
    "screenplay_summarize": ScreenplaySummarize,
    "gutenberg_extraction": GutenbergExtraction,
    "gutenberg_backtranslation": GutenbergBacktranslation,
    "gutenberg_backtranslation_from_txt": GutenbergBacktranslationFromTxt,
    "gutenberg_followup": GutenbergFollowUp,
    "writing_reward": WritingRewardAnnotate,
    "writing_score": WritingScoreAnnotate,
}


ALL_ENVIRONMENTS: dict[str, type[TextEnv]] = {
    "twenty_questions": TwentyQuestionsPolicyEnvironment,
}


async def run_environments(
    envs: List[TextEnv],
    n_epochs: int,
    save_every_n_batches: int,
    dataset_root_path: str,
):
    out_convs = []
    out_metadata = []

    logger.info(f"Running {len(envs)} environments, {n_epochs} epochs")

    output_dataset_name, output_dataset_format = (
        envs[0].task.output_dataset_name,
        envs[0].task.output_dataset_format,
    )
    output_dataset = Dataset.from_dict(
        {
            "conversation": [],
            "metadata": [],
        }
    )

    for i in range(n_epochs):
        active_envs = copy(envs)
        for j, env in enumerate(active_envs):
            env.seed = j + i * len(envs)
            env.reset()

        while len(active_envs) > 0:
            try:
                tasks = [env.step() for env in active_envs]
                step_results = await asyncio.gather(*tasks)

                indices_to_remove = []

                for k, (env, done) in enumerate(zip(active_envs, step_results)):
                    if done:
                        out_convs.append(env.conversation)
                        out_metadata.append(env.run_metadata)
                        indices_to_remove.append(k)

                for k in sorted(indices_to_remove, reverse=True):
                    active_envs.pop(k)

                logger.info(f"Environments running: {len(active_envs)}")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during environment steps: {e}")
                continue

        if i % save_every_n_batches == 0:
            out_rows = {
                "conversation": out_convs,
                "metadata": out_metadata,
            }
            out_rows = dictl(out_rows)
            save_output_dataset(
                output_dataset,
                output_dataset_name,
                out_rows,
                output_dataset_format,
                dataset_root_path,
            )

    if out_convs:
        out_rows = {
            "conversation": out_convs,
            "metadata": out_metadata,
        }
        out_rows = dictl(out_rows)
        save_output_dataset(
            output_dataset,
            output_dataset_name,
            out_rows,
            output_dataset_format,
            dataset_root_path,
        )


def main(
    task_name: str | None = None,
    environment_name: str | None = None,
    save_every_n_batches: int = 5,
    batch_size: int = 4,
    restart: bool = False,
    resume_input_position: bool = True,
    model: RemoteModel = "mistral-small-3",
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
    assert not (task_name and environment_name), (
        "Only one of task_name or environment_name should be passed"
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    load_dotenv(".env")
    args_override: GenWrapperArgs | None = None
    generation_wrapper = get_generation_wrapper(model, args_override)
    environment: Optional[TextEnv] = None

    if task_name:
        task = ALL_TASKS[task_name]()
    else:
        assert environment_name, "Environment name must be passed"
        environment = ALL_ENVIRONMENTS[environment_name](generation_wrapper, 0)
        task = environment.task
        args_override = task.gen_wrapper_args_override

    output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})
    split = task.seed_data_split

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
    elif task.seed_data_format == DatasetFormat.NONE:
        input_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})
    else:
        assert input_dataset_location, (
            f"Input dataset location must be provided, but is {input_dataset_location}"
        )
        if task.seed_data_format == DatasetFormat.HF_DATASET:
            input_dataset = cast(
                Dataset, load_dataset(input_dataset_location, split=split)
            )
        elif task.seed_data_format == DatasetFormat.TSV:
            seed_data = pd.read_csv(
                os.path.join(dataset_root_path, f"{input_dataset_location}.tsv"),
                on_bad_lines="skip",
            )
            input_dataset = Dataset.from_pandas(seed_data)
        elif task.seed_data_format == DatasetFormat.PARQUET:
            input_dataset = Dataset.from_parquet(
                os.path.join(dataset_root_path, f"{input_dataset_location}.parquet")
            )  # type: ignore
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

    # Generation loop for generation tasks

    if not environment_name:
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
