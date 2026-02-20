"""CLI entrypoint for dataset generation tasks.

This module resolves a task, loads seed/output datasets, runs async generation,
and persists the resulting dataset in the task's configured output format.
"""

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
)
from synthetic_data.tasks import BaseTask, BaseTaskV1, RunMode
from synthetic_data.tasks.next_chapter import (
    GutenbergSummaryContinuation,
)
from synthetic_data.tasks.roleplaying import (
    RoleplayingGameMultiStepTask,
)
from synthetic_data.utils import DatasetFormat


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
    save_every_n_batches: int,
    min_concurrent_rows: int,
):
    """Run one task over an input dataset and return the updated output dataset.

    `BaseTaskV1` uses batched prompt/completion generation. `BaseTask` uses an
    episode loop (`initial_step` -> repeated `step`) per row.
    """

    generation_stats = {
        "rows_appended": 0,
        "metric_calls": 0,
        "metric_error_calls": 0,
        "metric_tokens": 0,
        "metric_duration_s": 0.0,
    }
    run_started_at = time.time()
    target_rows = max(1, len(input_dataset) * max(1, n_epochs))

    def _log_progress(stage: str, batches_written: int) -> None:
        if generation_stats["metric_calls"] == 0:
            return
        elapsed = max(0.001, time.time() - run_started_at)
        error_rate = (
            generation_stats["metric_error_calls"] / generation_stats["metric_calls"]
        )
        rows = generation_stats["rows_appended"]
        rows_per_min = rows / elapsed * 60.0
        pct = min(100.0, (rows / target_rows) * 100.0)
        logger.info(
            "{}: batches={}, rows={}/{}, progress={:.1f}%, calls={}, errors={}, error_rate={:.3f}, tokens={}, model_duration_s={:.2f}, rows_per_min={:.2f}".format(
                stage,
                batches_written,
                rows,
                target_rows,
                pct,
                generation_stats["metric_calls"],
                generation_stats["metric_error_calls"],
                error_rate,
                generation_stats["metric_tokens"],
                generation_stats["metric_duration_s"],
                rows_per_min,
            )
        )

    def _accumulate_generation_stats(rows: list[dict]) -> None:
        for row in rows:
            generation_stats["rows_appended"] += 1
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            metrics = (
                metadata.get("generation_metrics", [])
                if isinstance(metadata, dict)
                else []
            )
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                generation_stats["metric_calls"] += 1
                if metric.get("finish_reason") == "error":
                    generation_stats["metric_error_calls"] += 1
                generation_stats["metric_tokens"] += int(
                    metric.get("usage_tokens", 0) or 0
                )
                generation_stats["metric_duration_s"] += float(
                    metric.get("duration_s", 0.0) or 0.0
                )

    async def _append_rows(rows: list[dict]) -> None:
        nonlocal output_dataset
        if not rows:
            return
        _accumulate_generation_stats(rows)
        new_ds = Dataset.from_list(rows)
        if len(output_dataset) == 0:
            output_dataset = new_ds
        else:
            output_dataset = concatenate_datasets([output_dataset, new_ds])

    async def _run_base_task() -> None:
        """Producer/worker/writer pipeline for multi-step episode tasks."""
        nonlocal output_dataset
        max_concurrent = max(1, int(min_concurrent_rows))

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
                    # Execute full episode lifecycle for one input row.
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
                _log_progress("Generation progress", batches_written)
                if (
                    save_every_n_batches > 0
                    and batches_written % save_every_n_batches == 0
                ):
                    if task.output_dataset_format == DatasetFormat.HF_DATASET:
                        output_dataset.push_to_hub(task.output_dataset_name)
                    elif task.output_dataset_format == DatasetFormat.PARQUET:
                        output_dataset.to_parquet(
                            f"{task.dataset_root_path}/{task.output_dataset_name}.parquet"
                        )
                    if generation_stats["metric_calls"] > 0:
                        _log_progress("Generation stats checkpoint", batches_written)
                results_queue.task_done()

        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        writer_task = asyncio.create_task(writer())

        await producer_task
        await queue.join()
        await results_queue.join()
        await asyncio.gather(*worker_tasks)
        await writer_task

    assert isinstance(task, BaseTask)
    await _run_base_task()

    if generation_stats["metric_calls"] > 0:
        _log_progress("Generation stats final", generation_stats["rows_appended"])

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
    min_concurrent_rows: int = 2,
    restart: bool = False,
    n_epochs: int = 1,
    run_mode: RunMode = "cli",
    roleplaying_max_user_responses: int = 13,
    roleplaying_num_episodes: int = 100,
    **kwargs,
):
    """
    Main CLI flow:
    1. Resolve task and runtime configuration.
    2. Load existing output dataset unless `restart=True`.
    3. Load/trim input dataset.
    4. Run async generation loop.
    5. Persist final output.
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
            task = RoleplayingGameMultiStepTask(
                run_mode,
                max_user_responses=roleplaying_max_user_responses,
                num_episodes=roleplaying_num_episodes,
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
    os.makedirs(out_dir, exist_ok=True)

    output_dataset: Dataset = asyncio.run(
        run_task(
            task,
            input_dataset,
            output_dataset,
            n_epochs,
            save_every_n_batches,
            min_concurrent_rows,
        )
    )

    if task.output_dataset_format == DatasetFormat.HF_DATASET:
        output_dataset.push_to_hub(task.output_dataset_name)
    elif task.output_dataset_format == DatasetFormat.PARQUET:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{out_dir}/{task.output_dataset_name}.parquet"
        output_dataset.to_parquet(filename)
    else:
        raise ValueError(f"Unsupported output format: {task.output_dataset_format}")


if __name__ == "__main__":
    fire.Fire(main)
