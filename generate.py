import asyncio
from copy import copy
import time
from statistics import mean
from typing import Dict, Optional, cast, List
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
from synthetic_data.tasks import BaseTask, RunMode
from synthetic_data.tasks.roleplaying import RoleplayingGame, RoleplayingGameEnvironment
from synthetic_data.tasks.writing import (
    GutenbergBacktranslationFromTxt,
    GutenbergExtraction,
    GutenbergBacktranslation,
    ScreenplaySummarize,
    GenerationBestOfN,
)
from synthetic_data.utils import DatasetFormat, print_result_dicts
from gyms import TwentyQuestionsPolicyEnvironment


TaskName = Literal[
    "screenplay_summarize",
    "gutenberg_extraction",
    "gutenberg_backtranslation",
    "gutenberg_backtranslation_from_txt",
    "generation_best_of_n",
    "roleplaying_game",
]
ALL_TASKS: Dict[TaskName, type[BaseTask]] = {
    "screenplay_summarize": ScreenplaySummarize,
    "gutenberg_extraction": GutenbergExtraction,
    "gutenberg_backtranslation": GutenbergBacktranslation,
    "gutenberg_backtranslation_from_txt": GutenbergBacktranslationFromTxt,
    "generation_best_of_n": GenerationBestOfN,
    "roleplaying_game": RoleplayingGame,
}


ALL_ENVIRONMENTS: dict[str, type[TextEnv]] = {
    "twenty_questions": TwentyQuestionsPolicyEnvironment,
    "roleplaying_game": RoleplayingGameEnvironment,
}


class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=1.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0

    def update(self, current_value: float, dt: float = 1.0) -> float:
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.previous_error = error
        return output


class SimplePIDBatchController:
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        target_throughput: float = 2.0,
    ):
        self._current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_throughput = target_throughput

        # PID controller for throughput optimization
        self.pid_controller = PIDController(
            kp=2.0, ki=0.1, kd=0.5, setpoint=target_throughput
        )

        # Simple throughput tracking
        self.throughput_history: List[float] = []
        self.history_window = 5

    def record_batch_performance(
        self, batch_size: int, batch_time: float, num_items: int
    ):
        """Record performance and adjust batch size using PID controller"""
        throughput = num_items / batch_time if batch_time > 0 else 0
        self.throughput_history.append(throughput)

        # Keep only recent history
        if len(self.throughput_history) > self.history_window:
            self.throughput_history = self.throughput_history[-self.history_window :]

        # Only adjust after we have some data
        if len(self.throughput_history) >= 3:
            current_throughput = mean(self.throughput_history)
            pid_output = self.pid_controller.update(current_throughput)

            # Convert PID output to batch size adjustment
            adjustment = int(pid_output)
            new_batch_size = self._current_batch_size + adjustment

            # Apply constraints
            new_batch_size = max(
                self.min_batch_size, min(self.max_batch_size, new_batch_size)
            )

            if new_batch_size != self._current_batch_size:
                logger.info(
                    f"PID batch size adjustment: {self._current_batch_size} → {new_batch_size} "
                    f"(throughput: {current_throughput:.2f} vs target: {self.target_throughput})"
                )
                self._current_batch_size = new_batch_size

    def get_current_batch_size(self) -> int:
        return self._current_batch_size


async def process_batch(
    task: BaseTask, generation_wrapper, input_rows: list[dict]
) -> tuple[list[dict], float]:
    logger.info(f"Processing batch of {len(input_rows)} rows")
    start_time = time.time()
    output_rows = await task.generate(generation_wrapper, input_rows)
    batch_time = time.time() - start_time

    if len(output_rows) == 0:
        logger.warning("Skipping empty batch")
        return [], batch_time
    return output_rows, batch_time


async def process_dataset(
    task: BaseTask,
    generation_wrapper,
    input_dataset: Dataset,
    initial_batch_size: int,
    save_every_n_batches: int,
    output_dataset: Dataset,
    out_dir: str,
) -> list[dict]:
    all_new_dataset_rows = []
    batch_count = 0

    # Initialize simple PID batch size controller (always enabled)
    batch_controller = SimplePIDBatchController(
        initial_batch_size=initial_batch_size,
        min_batch_size=1,
        max_batch_size=min(64, initial_batch_size * 4),
        target_throughput=2.0,
    )

    i = 0
    while i < len(input_dataset):
        current_batch_size = batch_controller.get_current_batch_size()
        batch = input_dataset[i : i + current_batch_size]
        if not isinstance(batch, list):
            batch = [batch]

        preprocessed_rows = []
        for row in batch:
            result = await task.preprocess_row(row)
            if result:
                preprocessed_rows.extend(result)

        if preprocessed_rows:
            output_rows, batch_time = await process_batch(
                task, generation_wrapper, preprocessed_rows
            )

            # Record performance metrics for autoscaling
            batch_controller.record_batch_performance(
                current_batch_size, batch_time, len(preprocessed_rows)
            )

            # Log throughput metrics
            if batch_time > 0:
                throughput = len(preprocessed_rows) / batch_time
                logger.info(
                    f"Batch throughput: {throughput:.2f} items/sec, "
                    f"batch_size: {current_batch_size}, time: {batch_time:.1f}s"
                )

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
                    out_dir,
                )

        i += current_batch_size

    if batch_controller.throughput_history:
        avg_throughput = mean(batch_controller.throughput_history)
        logger.info(f"Final average throughput: {avg_throughput:.2f} items/sec")
        logger.info(f"Final batch size: {batch_controller.get_current_batch_size()}")

    return all_new_dataset_rows


def main(
    task_name: TaskName | None = None,
    environment_name: str | None = None,
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
        task = ALL_TASKS[task_name](run_mode)
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
                    out_dir,
                )
            )

            # Final save of any remaining rows
            if all_new_dataset_rows:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    all_new_dataset_rows,
                    task.output_dataset_format,
                    out_dir,
                )
    else:
        assert environment, "Environment must be passed"
        envs = [copy(environment) for _ in range(batch_size)]
        asyncio.run(
            run_environments(
                envs,
                n_epochs,
                save_every_n_batches,
                run_mode,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
