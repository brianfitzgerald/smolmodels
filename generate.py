import asyncio
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, cast

import fire
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger

from synthetic_data.generation import (
    GenerationWrapper,
    GenWrapperArgs,
    RemoteModel,
    get_generation_wrapper,
    save_output_dataset,
)
from synthetic_data.tasks import BaseTask, BaseTaskV1, RunMode
from synthetic_data.tasks.roleplaying import (
    RoleplayingGameMultiStepTask,
)
from synthetic_data.utils import DatasetFormat


@dataclass
class ThroughputConfig:
    """Configuration for throughput control"""

    target_tokens_per_second: float = 1000.0  # Target throughput in tokens/sec
    max_tokens_per_second: float = 2000.0  # Maximum allowed throughput
    pid_kp: float = 0.1  # Proportional gain
    pid_ki: float = 0.01  # Integral gain
    pid_kd: float = 0.05  # Derivative gain
    measurement_window: float = 10.0  # Window for measuring throughput (seconds)
    min_concurrent_requests: int = 1  # Minimum concurrent requests
    max_concurrent_requests: int = 50  # Maximum concurrent requests


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


class ThroughputMonitor:
    """Monitors throughput for different model providers"""

    def __init__(self, config: ThroughputConfig):
        self.config = config
        self.provider_throughput = defaultdict(lambda: deque(maxlen=100))
        self.provider_request_counts = defaultdict(int)
        self.provider_token_counts = defaultdict(int)
        self.last_reset_time = time.time()
        self.pid_controllers = defaultdict(
            lambda: PIDController(config.pid_kp, config.pid_ki, config.pid_kd)
        )

    def record_request(self, provider: str, tokens: int, duration: float):
        """Record a completed request"""
        current_time = time.time()

        # Reset counters if measurement window has passed
        if current_time - self.last_reset_time > self.config.measurement_window:
            self.provider_request_counts.clear()
            self.provider_token_counts.clear()
            self.last_reset_time = current_time

        # Record throughput measurement
        if duration > 0:
            throughput = tokens / duration
            self.provider_throughput[provider].append((current_time, throughput))

        # Update counters
        self.provider_request_counts[provider] += 1
        self.provider_token_counts[provider] += tokens

    def get_current_throughput(self, provider: str) -> float:
        """Get current throughput for a provider"""
        current_time = time.time()
        window_start = current_time - self.config.measurement_window

        # Filter measurements within the window
        recent_measurements = [
            (t, tp) for t, tp in self.provider_throughput[provider] if t >= window_start
        ]

        if not recent_measurements:
            return 0.0

        # Calculate average throughput over the window
        total_tokens = sum(
            tp * self.config.measurement_window for _, tp in recent_measurements
        )
        return total_tokens / self.config.measurement_window

    def get_optimal_concurrency(self, provider: str) -> int:
        """Get optimal concurrency level using PID control"""
        current_throughput = self.get_current_throughput(provider)

        # Get PID control output
        pid_output = self.pid_controllers[provider].update(
            self.config.target_tokens_per_second, current_throughput
        )

        # Convert PID output to concurrency level
        # Assume linear relationship between concurrency and throughput
        base_concurrency = max(1, int(pid_output))

        # Clamp to valid range
        concurrency = max(
            self.config.min_concurrent_requests,
            min(self.config.max_concurrent_requests, base_concurrency),
        )

        logger.debug(
            f"Provider {provider}: current_throughput={current_throughput:.1f}, "
            f"target={self.config.target_tokens_per_second:.1f}, "
            f"concurrency={concurrency}"
        )

        return concurrency


@dataclass
class WrapperInfo:
    """Information about a registered wrapper"""

    wrapper: GenerationWrapper
    model_name: str
    provider: str
    max_concurrent: Optional[int] = None


class AutoscalingGenerationManager:
    """Manages multiple generation wrappers with autoscaling"""

    def __init__(self, config: ThroughputConfig):
        self.config = config
        self.throughput_monitor = ThroughputMonitor(config)
        self.wrapper_registry: Dict[GenerationWrapper, WrapperInfo] = {}

    def add_wrapper(
        self,
        model: str,
        wrapper: GenerationWrapper,
        max_concurrent: Optional[int] = None,
    ) -> None:
        """Add a generation wrapper to the manager"""

        # Create wrapper info
        wrapper_info = WrapperInfo(
            wrapper=wrapper,
            model_name=model,
            provider=wrapper.provider_name,
            max_concurrent=max_concurrent,
        )

        # Register the wrapper
        self.wrapper_registry[wrapper] = wrapper_info

        # Override max_concurrent if specified
        if max_concurrent is not None:
            wrapper.set_max_concurrent(max_concurrent)

    def get_wrapper_info(self, wrapper: GenerationWrapper) -> WrapperInfo:
        """Get information about a registered wrapper"""
        if wrapper not in self.wrapper_registry:
            raise ValueError(f"Wrapper {wrapper} is not registered in the manager")
        return self.wrapper_registry[wrapper]

    def get_all_wrappers(self) -> List[GenerationWrapper]:
        """Get all registered wrappers"""
        return list(self.wrapper_registry.keys())

    def get_wrappers_by_provider(self, provider: str) -> List[GenerationWrapper]:
        """Get all wrappers for a specific provider"""
        return [
            wrapper
            for wrapper, info in self.wrapper_registry.items()
            if info.provider == provider
        ]

    async def generate_with_autoscaling(
        self, wrapper: GenerationWrapper, conversations: List[Any], **kwargs: Any
    ) -> List[str]:
        """Generate with autoscaling based on throughput"""
        wrapper_info = self.get_wrapper_info(wrapper)
        provider = wrapper_info.provider

        # Get optimal concurrency
        optimal_concurrency = self.throughput_monitor.get_optimal_concurrency(provider)

        wrapper.set_max_concurrent(optimal_concurrency)

        # Execute generation with timing
        start_time = time.time()
        results = await wrapper.generate(conversations, **kwargs)
        end_time = time.time()

        # Calculate tokens (rough estimate)
        total_tokens = sum(
            len(str(conv)) // 4 for conv in conversations
        )  # Rough token estimation
        total_tokens += sum(len(str(result)) // 4 for result in results)

        # Record throughput
        duration = end_time - start_time
        self.throughput_monitor.record_request(provider, total_tokens, duration)

        return results


async def run_task(
    task: BaseTask,
    generation_wrapper,
    input_dataset: Dataset,
    output_dataset: Dataset,
    batch_size: int,
    n_epochs: int,
    save_every_n_batches: int,
    out_dir: str,
    throughput_config: ThroughputConfig | None = None,
):
    """
    Run a task with autoscaling throughput control.

    Args:
        task: The task to run
        generation_wrapper: The generation wrapper to use
        input_dataset: Input dataset
        output_dataset: Output dataset to append to
        batch_size: Batch size for processing
        n_epochs: Number of epochs to run
        save_every_n_batches: Save every N batches
        out_dir: Output directory for saving
        throughput_config: Configuration for throughput control
    """
    if throughput_config is None:
        throughput_config = ThroughputConfig()

    # Initialize autoscaling manager
    autoscaling_manager = AutoscalingGenerationManager(throughput_config)

    # Add the main generation wrapper
    model_name = getattr(generation_wrapper, "model_name", "unknown")
    autoscaling_manager.add_wrapper(model_name, generation_wrapper)

    # For multi-step tasks, add additional wrappers from the task's generation_wrappers dict
    if hasattr(task, "generation_wrappers"):
        # This is a multi-step task with multiple models
        for model_name, wrapper in task.generation_wrappers.items():
            if (
                model_name != "generation"
            ):  # Skip the main generation wrapper as it's already added
                autoscaling_manager.add_wrapper(model_name, wrapper)

    logger.info("Starting task with autoscaling throughput control")
    logger.info(
        f"Target throughput: {throughput_config.target_tokens_per_second} tokens/sec"
    )
    logger.info(f"Output dataset size: {len(output_dataset)}")

    total_processed = 0
    batch_count = 0

    for epoch in range(n_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{n_epochs}")

        # Process dataset in batches
        for i in range(0, len(input_dataset), batch_size):
            batch_end = min(i + batch_size, len(input_dataset))
            batch_data = input_dataset[i:batch_end]

            logger.info(f"Processing batch {batch_count + 1}: rows {i}-{batch_end - 1}")

            results = await process_multi_step_task(
                task, autoscaling_manager, batch_data
            )
            # Add results to output dataset
            if results:
                new_dataset = Dataset.from_list(results)
                output_dataset = concatenate_datasets([output_dataset, new_dataset])
                total_processed += len(results)

            batch_count += 1

            # Save periodically
            if batch_count % save_every_n_batches == 0:
                logger.info(f"Saving checkpoint after {batch_count} batches")
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    [],  # No new rows to add, just save current state
                    task.output_dataset_format,
                    out_dir,
                )

    logger.info(f"Task completed. Total processed: {total_processed}")

    # Final save
    save_output_dataset(
        output_dataset,
        task.output_dataset_name,
        [],  # No new rows to add, just save current state
        task.output_dataset_format,
        out_dir,
    )

    return output_dataset


async def process_single_step_task(
    task: BaseTaskV1,
    autoscaling_manager: AutoscalingGenerationManager,
    batch_data: Any,
) -> List[Dict[str, Any]]:
    """Process a single-step task"""
    # Preprocess batch
    processed_batches = []
    for row in batch_data:
        # Convert row to dict if it's not already
        row_dict = dict(row) if not isinstance(row, dict) else row
        processed = await task.preprocess_row(row_dict)
        processed_batches.extend(processed)

    if not processed_batches:
        return []

    # Format conversations
    conversations = task.format_input_conversation(processed_batches)

    if not conversations:
        return []

    # Generate with autoscaling
    # Use the main generation wrapper (first registered wrapper)
    all_wrappers = autoscaling_manager.get_all_wrappers()
    if not all_wrappers:
        raise ValueError("No wrappers registered in autoscaling manager")
    main_wrapper = all_wrappers[0]

    completions = await autoscaling_manager.generate_with_autoscaling(
        main_wrapper, conversations
    )

    # Format output rows
    return task.format_output_rows(completions, processed_batches)


async def process_multi_step_task(
    task: BaseTask,
    autoscaling_manager: AutoscalingGenerationManager,
    batch_data: Any,
) -> List[Dict[str, Any]]:
    """Process a multi-step task (like roleplaying)"""
    results = []

    for i, row in enumerate(batch_data):
        # Create new episode - use the main generation wrapper
        all_wrappers = autoscaling_manager.get_all_wrappers()
        if not all_wrappers:
            raise ValueError("No wrappers registered in autoscaling manager")

        # Set the generation wrapper in the episode
        episode = await task.start_episode(row)

        # Run episode steps until completion
        while True:
            step_result = await task.step_episode(episode)
            if (
                step_result
            ):  # If step_episode returns non-empty list, episode is complete
                break

        # Get output row
        output_rows = task.get_output_row(episode)
        results.extend(output_rows)

    return results


TaskName = Literal[
    "screenplay_summarize",
    "gutenberg_extraction",
    "gutenberg_backtranslation",
    "gutenberg_backtranslation_from_txt",
    "generation_best_of_n",
    "roleplaying_game",
]
ALL_TASKS: Dict[TaskName, type[BaseTask]] = {
    "roleplaying_game": RoleplayingGameMultiStepTask,
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
