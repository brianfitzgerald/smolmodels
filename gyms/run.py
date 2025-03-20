import asyncio
from copy import copy
import traceback
from typing import List

from datasets import Dataset
from loguru import logger

from gyms.twenty_questions.env import TextEnv
from synthetic_data.generation import save_output_dataset
from synthetic_data.utils import dictl


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
