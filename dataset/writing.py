import asyncio
from functools import partial
import math
import statistics

from loguru import logger
from synthetic_data.generation import (
    GenerationWrapper,
    get_generation_wrapper,
)
from synthetic_data.writing_judge import CreativeWritingBench
from trl_wrapper.wrapper_config import SmDataset
from datasets import Dataset
from typing import Coroutine, Optional, TypeVar
import pandas as pd
from trl.trainer.grpo_trainer import RewardFunc
from synthetic_data.tasks.writing import score_writing
import threading
from model.reasoning import logger_reward
from typing import Any


def _get_scores(score_dicts: list[dict[str, float | None]]) -> list[dict[str, float]]:
    all_keys_union = set()
    score_totals = []
    for d in score_dicts:
        all_keys_union.update(d.keys())
    for d in score_dicts:
        total = 0.0
        for k in all_keys_union:
            if k in d and d.get(k) is not None:
                total += d[k] or 0.0
        score_totals.append(total)
    return score_totals


def _pick_dpo_pair(group: pd.Series) -> pd.Series:
    group["score_total"] = _get_scores(group["scores"])
    chosen_row = group.loc[group["score_total"].idxmax()]
    rejected_row = group.loc[group["score_total"].idxmin()]
    return pd.Series(
        {
            "chosen": chosen_row["completion"],
            "rejected": rejected_row["completion"],
            "score_chosen": chosen_row["score_total"],
            "score_rejected": rejected_row["score_total"],
            "model_chosen": chosen_row["model_id"],
            "model_rejected": rejected_row["model_id"],
            "prompt": chosen_row["instruction"],
        }
    )


class WritingDPODataModule(SmDataset):
    def setup(self, stage: Optional[str] = None):
        dataset_pd = pd.read_parquet("../dataset_files/backtranslate_best_of_n.parquet")

        dataset_pd["instruction_id"] = (
            dataset_pd["instruction"].astype("category").cat.codes
        )

        dataset_pd = (
            dataset_pd.groupby("instruction_id").apply(_pick_dpo_pair).reset_index()  # type: ignore
        )
        dataset = Dataset.from_pandas(dataset_pd).train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]


def _map_writing_grpo(example: dict) -> dict:
    # TODO maybe compare to existing rewards?
    return {
        "prompt": [
            {"role": "user", "content": example["instruction"]},
        ],
    }


T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    # Running in an environment like Jupyter Notebook.
    if loop is not None and loop.is_running():
        result_container = []

        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(coro)
            new_loop.close()
            result_container.append(result)

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result_container[0]
    # No event loop running; use asyncio.run for a clean execution.
    else:
        return asyncio.run(coro)


MAX_SCORE = 20


def llm_judge_func(
    prompts: list[str],
    completions: list[str],
    generation_wrapper: GenerationWrapper | None = None,
    bench: CreativeWritingBench | None = None,
) -> list[float]:
    async def run_score():
        assert generation_wrapper is not None
        assert bench is not None
        out = await score_writing(completions, prompts, bench, generation_wrapper)
        return out

    scores = run_sync(run_score())

    score_vals = []
    for s in scores:
        total = 0.0
        for v in s.values():
            v = max(min(v, MAX_SCORE), 0.0)
            total += v / MAX_SCORE
        avg_score = round(total / len(s), 2)
        score_vals.append(avg_score)
    return score_vals


def create_llm_judge_func(
    generation_wrapper: GenerationWrapper, bench: CreativeWritingBench, **kwargs
) -> RewardFunc:
    """Create a function for LLM judging with a proper name."""

    def named_llm_judge_func(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        return llm_judge_func(prompts, completions, generation_wrapper, bench, **kwargs)

    named_llm_judge_func.__name__ = (
        f"llm_judge_func_{generation_wrapper.__class__.__name__}"
    )
    return named_llm_judge_func


class WritingGRPODataModule(SmDataset):
    """Online rewarding using generation wrappers."""

    def setup(self, stage: Optional[str] = None):
        dataset_pd = pd.read_parquet(
            "../dataset_files/gutenberg_backtranslate_from_txt.parquet"
        )

        dataset = Dataset.from_pandas(dataset_pd).train_test_split(test_size=0.1)
        dataset = dataset.map(_map_writing_grpo)
        dataset = dataset.remove_columns(
            [
                "paragraph",
                "text",
                "title",
                "author",
                "id",
                "instruction",
            ]
        )
        dataset.shuffle(seed=42)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.generation_wrapper = get_generation_wrapper("gemini-2.0-flash")
        self.bench = CreativeWritingBench(self.config.run_mode)

    def reward_functions(self) -> list[RewardFunc]:
        return [
            create_llm_judge_func(self.generation_wrapper, self.bench),
            logger_reward,
        ]
