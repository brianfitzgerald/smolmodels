from __future__ import annotations

import asyncio
from copy import copy
from statistics import mean
from typing import Any, List
import time

from datasets import Dataset
from loguru import logger

from synthetic_data.generation import GenerationWrapper, save_output_dataset
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import print_result_dicts
from datasets import Dataset


class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
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
        self.pid_controller = PIDController(kp=2.0, ki=0.1, kd=0.5, setpoint=target_throughput)
        self.throughput_history: List[float] = []
        self.history_window = 5

    def record_batch_performance(self, batch_size: int, batch_time: float, num_items: int):
        throughput = num_items / batch_time if batch_time > 0 else 0
        self.throughput_history.append(throughput)
        if len(self.throughput_history) > self.history_window:
            self.throughput_history = self.throughput_history[-self.history_window :]
        if len(self.throughput_history) >= 3:
            current_throughput = mean(self.throughput_history)
            pid_output = self.pid_controller.update(current_throughput)
            adjustment = int(pid_output)
            new_batch_size = self._current_batch_size + adjustment
            new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
            if new_batch_size != self._current_batch_size:
                logger.info(
                    f"PID batch size adjustment: {self._current_batch_size} → {new_batch_size} (throughput: {current_throughput:.2f} vs target: {self.target_throughput})"
                )
                self._current_batch_size = new_batch_size

    def get_current_batch_size(self) -> int:
        return self._current_batch_size


async def _process_batch(task: BaseTask, generation_wrapper: GenerationWrapper, input_rows: list[dict]) -> tuple[list[dict], float]:
    logger.info(f"Processing batch of {len(input_rows)} rows")
    start_time = time.time()
    output_rows = await task.generate(generation_wrapper, input_rows)
    batch_time = time.time() - start_time
    if len(output_rows) == 0:
        logger.warning("Skipping empty batch")
        return [], batch_time
    return output_rows, batch_time


async def _process_dataset(
    task: BaseTask,
    generation_wrapper: GenerationWrapper,
    input_dataset: Dataset,
    initial_batch_size: int,
    save_every_n_batches: int,
    output_dataset: Dataset,
    out_dir: str,
) -> list[dict]:
    all_new_dataset_rows: list[dict] = []
    batch_count = 0
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

        preprocessed_rows: list[dict] = []
        for row in batch:
            result = await task.preprocess_row(row)
            if result:
                preprocessed_rows.extend(result)

        if preprocessed_rows:
            output_rows, batch_time = await _process_batch(task, generation_wrapper, preprocessed_rows)
            batch_controller.record_batch_performance(current_batch_size, batch_time, len(preprocessed_rows))
            if batch_time > 0:
                throughput = len(preprocessed_rows) / batch_time
                logger.info(
                    f"Batch throughput: {throughput:.2f} items/sec, batch_size: {current_batch_size}, time: {batch_time:.1f}s"
                )
            if output_rows:
                print_result_dicts(output_rows)
                all_new_dataset_rows.extend(output_rows)
            batch_count += 1
            if batch_count % save_every_n_batches == 0:
                save_output_dataset(output_dataset, task.output_dataset_name, all_new_dataset_rows, task.output_dataset_format, out_dir)
                all_new_dataset_rows = []
        i += current_batch_size

    if batch_controller.throughput_history:
        avg_throughput = mean(batch_controller.throughput_history)
        logger.info(f"Final average throughput: {avg_throughput:.2f} items/sec")
        logger.info(f"Final batch size: {batch_controller.get_current_batch_size()}")

    return all_new_dataset_rows


async def run_task(
    task: BaseTask,
    generation_wrapper: GenerationWrapper,
    input_dataset: Dataset,
    output_dataset: Dataset,
    batch_size: int,
    n_epochs: int,
    save_every_n_batches: int,
    dataset_root_path: str,
):
    """Unified runner for both single-step and multi-step tasks."""
    # Probe if episodes are implemented
    episodes_supported = False
    try:
        _ = task.new_episode(generation_wrapper, 0)
        episodes_supported = True
    except NotImplementedError:
        episodes_supported = False
    except Exception:
        # If method exists and errors due to missing resources, assume episodes path
        episodes_supported = True

    if not episodes_supported:
        # Use single-step dataset processing
        for _ in range(n_epochs):
            new_rows = await _process_dataset(
                task,
                generation_wrapper,
                input_dataset,
                batch_size,
                save_every_n_batches,
                output_dataset,
                dataset_root_path,
            )
            if new_rows:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    new_rows,
                    task.output_dataset_format,
                    dataset_root_path,
                )
        return

    # Episodes path (multi-step)
    logger.info(
        f"Running episodic task {task.output_dataset_name}: batch={batch_size}, epochs={n_epochs}"
    )
    out_rows: List[dict] = []
    for epoch in range(n_epochs):
        seeds = [epoch * batch_size + i for i in range(batch_size)]
        episodes = [task.new_episode(generation_wrapper, s) for s in seeds]
        active = copy(episodes)
        while active:
            dones = await asyncio.gather(
                *[task.step_episode(generation_wrapper, ep) for ep in active]
            )
            new_active = []
            for ep, done in zip(active, dones):
                if done:
                    out_rows.append(task.get_output_row(ep))
                else:
                    new_active.append(ep)
            active = new_active

            if len(out_rows) and len(out_rows) % (save_every_n_batches * batch_size) == 0:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    out_rows,
                    task.output_dataset_format,
                    dataset_root_path,
                )
                out_rows = []
    if out_rows:
        save_output_dataset(
            output_dataset,
            task.output_dataset_name,
            out_rows,
            task.output_dataset_format,
            dataset_root_path,
        )
