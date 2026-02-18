import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, cast

import fire
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    RemoteModel,
    get_generation_wrapper,
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
    task: BaseTask | BaseTaskV1,
    input_dataset: Dataset,
    output_dataset: Dataset,
    n_epochs: int,
    batch_size: int,
    save_every_n_batches: int,
    model: RemoteModel,
):
    async def _append_rows(rows: list[dict]) -> None:
        nonlocal output_dataset
        if not rows:
            return
        new_ds = Dataset.from_list(rows)
        if len(output_dataset) == 0:
            output_dataset = new_ds
        else:
            output_dataset = concatenate_datasets([output_dataset, new_ds])

    async def _run_base_task_v1() -> None:
        nonlocal output_dataset
        generation_wrapper = get_generation_wrapper(
            model, task.gen_wrapper_args_override
        )
        max_concurrent = generation_wrapper.gen_wrapper_args.max_concurrent
        max_batch = max(1, batch_size)
        queue: asyncio.Queue[dict | None] = asyncio.Queue(
            maxsize=max_concurrent * max_batch * 2
        )
        results_queue: asyncio.Queue[list[dict] | None] = asyncio.Queue()

        async def producer() -> None:
            for _ in range(n_epochs):
                for row in input_dataset:
                    processed_rows = await task.preprocess_row(row)
                    for processed in processed_rows:
                        await queue.put(processed)
            for _ in range(max_concurrent):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                batch = [item]
                while len(batch) < max_batch:
                    try:
                        next_item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if next_item is None:
                        await queue.put(None)
                        break
                    batch.append(next_item)
                try:
                    results = await task.generate(generation_wrapper, batch)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    results = []
                await results_queue.put(results)
                for _ in range(len(batch)):
                    queue.task_done()

            await results_queue.put(None)

        async def writer() -> None:
            batches_written = 0
            finished_workers = 0
            while finished_workers < max_concurrent:
                results = await results_queue.get()
                if results is None:
                    finished_workers += 1
                    results_queue.task_done()
                    continue
                await _append_rows(results)
                batches_written += 1
                if (
                    save_every_n_batches > 0
                    and batches_written % save_every_n_batches == 0
                ):
                    if task.output_dataset_format == DatasetFormat.HF_DATASET:
                        output_dataset.push_to_hub(task.output_dataset_name)
                    elif task.output_dataset_format == DatasetFormat.PARQUET:
                        output_dataset.to_parquet(f"{task.output_dataset_name}.parquet")
                results_queue.task_done()

        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        writer_task = asyncio.create_task(writer())

        await producer_task
        await queue.join()
        await results_queue.join()
        await asyncio.gather(*worker_tasks)
        await writer_task

    async def _run_base_task() -> None:
        nonlocal output_dataset
        max_concurrent = 1
        if task.generation_wrappers:
            max_concurrent = max(
                w.gen_wrapper_args.max_concurrent
                for w in task.generation_wrappers.values()
            )

        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=max_concurrent * 2)
        results_queue: asyncio.Queue[list[dict] | None] = asyncio.Queue()

        async def producer() -> None:
            for _ in range(n_epochs):
                for row in input_dataset:
                    await queue.put(row)
            for _ in range(max_concurrent):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                try:
                    episode = await task.initial_step(item)  # type: ignore[arg-type]
                    finished = False
                    while not finished:
                        episode, finished = await task.step(episode)  # type: ignore[arg-type]
                    row = task.format_episode(episode)  # type: ignore[arg-type]
                    await results_queue.put([row])
                except Exception as e:
                    logger.error(f"Error processing episode: {e}")
                    await results_queue.put([])
                queue.task_done()
            await results_queue.put(None)

        async def writer() -> None:
            batches_written = 0
            finished_workers = 0
            while finished_workers < max_concurrent:
                results = await results_queue.get()
                if results is None:
                    finished_workers += 1
                    results_queue.task_done()
                    continue
                await _append_rows(results)
                batches_written += 1
                if (
                    save_every_n_batches > 0
                    and batches_written % save_every_n_batches == 0
                ):
                    if task.output_dataset_format == DatasetFormat.HF_DATASET:
                        output_dataset.push_to_hub(task.output_dataset_name)
                    elif task.output_dataset_format == DatasetFormat.PARQUET:
                        output_dataset.to_parquet(f"{task.output_dataset_name}.parquet")
                results_queue.task_done()

        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        writer_task = asyncio.create_task(writer())

        await producer_task
        await queue.join()
        await results_queue.join()
        await asyncio.gather(*worker_tasks)
        await writer_task

    if isinstance(task, BaseTaskV1):
        await _run_base_task_v1()
    else:
        await _run_base_task()

    return output_dataset


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
    roleplaying_max_user_responses: int | None = None,
    roleplaying_num_episodes: int | None = None,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    if "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token, add_to_git_credential=True)

    if task_name is None:
        raise ValueError("`task_name` must be provided")

    if task_name in ALL_TASKS:
        if task_name == "roleplaying_game":
            max_user_responses = (
                roleplaying_max_user_responses
                if roleplaying_max_user_responses is not None
                else 10
            )
            num_episodes = (
                roleplaying_num_episodes
                if roleplaying_num_episodes is not None
                else 1000
            )
            task = RoleplayingGameMultiStepTask(
                run_mode,
                max_user_responses=max_user_responses,
                num_episodes=num_episodes,
            )
        else:
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

    if len(output_dataset) > 0 and not restart:
        shared_cols = [
            col
            for col in ["id", "prompt_id", "instruction", "Prompt", "text"]
            if col in input_dataset.column_names and col in output_dataset.column_names
        ]
        if shared_cols:
            key = shared_cols[0]
            existing = set(output_dataset[key])
            input_dataset = input_dataset.filter(lambda x: x[key] not in existing)
            logger.info(
                f"Filtered input dataset using key '{key}': {len(input_dataset)} rows remaining"
            )
        else:
            skip_n = min(len(output_dataset), len(input_dataset))
            if skip_n > 0:
                input_dataset = input_dataset.select(range(skip_n, len(input_dataset)))
                logger.info(
                    f"Skipped first {skip_n} rows based on existing output length"
                )

    out_dir = (
        "../dataset_files"
        if run_mode == "notebook"
        else "./dataset_files"
        if run_mode == "cli"
        else "/model-weights/dataset_files"
    )

    output_dataset = asyncio.run(
        run_task(
            task,  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
            input_dataset,
            output_dataset,
            n_epochs,
            batch_size,
            save_every_n_batches,
            model,
        )
    )

    if task.output_dataset_format == DatasetFormat.HF_DATASET:
        output_dataset.push_to_hub(task.output_dataset_name)
    elif task.output_dataset_format == DatasetFormat.PARQUET:
        filename = f"{out_dir}/{task.output_dataset_name}.parquet"
        output_dataset.to_parquet(filename)
    else:
        raise ValueError(f"Unsupported output format: {task.output_dataset_format}")


if __name__ == "__main__":
    fire.Fire(main)
