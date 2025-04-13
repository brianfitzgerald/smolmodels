import asyncio
from functools import partial
from synthetic_data.generation import (
    GenerationWrapper,
    get_generation_wrapper,
)
from synthetic_data.writing_judge import CreativeWritingBench
from trl_wrapper.wrapper_config import SmDataset
from datasets import Dataset
from typing import Optional
import pandas as pd
from trl.trainer.grpo_trainer import RewardFunc
from synthetic_data.tasks.writing import score_writing


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


def llm_judge_func(
    prompts: list,
    completions: list,
    generation_wrapper: GenerationWrapper,
    bench: CreativeWritingBench,
    **kwargs,
) -> list[float]:
    # https://stackoverflow.com/questions/55409641/asyncio-run-cannot-be-called-from-a-running-event-loop-when-using-jupyter-no
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    scores = []

    def _task_done(t):
        result = t.result()
        print(f"Result: {result}")
        scores.extend([sum(score.values()) for score in result])

    if loop and loop.is_running():
        tsk = loop.create_task(
            score_writing(completions, prompts, bench, generation_wrapper)
        )
        tsk.add_done_callback(_task_done)
    else:
        print("Starting new event loop")
        scores = asyncio.run(
            score_writing(completions, prompts, bench, generation_wrapper)
        )
    print(f"Scores: {scores}")
    score_sums = [sum(score.values()) for score in scores]
    print(f"Score sums: {score_sums}")
    return score_sums


def _map_writing_grpo(example: dict) -> dict:
    # TODO maybe compare to existing rewards?
    return {
        "prompt": [
            {"role": "user", "content": example["instruction"]},
        ],
    }


def _wrap_llm_judge_func(
    generation_wrapper: GenerationWrapper, bench: CreativeWritingBench
):
    """Wrapper function that preserves __name__ attribute for the reward function."""

    def wrapped_func(prompts: list, completions: list, **kwargs) -> list[float]:
        return llm_judge_func(
            prompts, completions, generation_wrapper=generation_wrapper, bench=bench
        )

    wrapped_func.__name__ = "llm_judge_func"
    return wrapped_func


class WritingGRPODataModule(SmDataset):
    """Online rewarding using generation wrappers."""

    def setup(self, stage: Optional[str] = None):
        dataset_pd = pd.read_parquet(
            "../dataset_files/gutenberg_backtranslate_from_txt.parquet"
        )

        dataset = Dataset.from_pandas(dataset_pd).train_test_split(test_size=0.1)
        dataset = dataset.map(_map_writing_grpo)
        dataset.shuffle(seed=42)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.generation_wrapper = get_generation_wrapper("gemini-2.0-flash")
        self.bench = CreativeWritingBench(self.config.run_mode)

    def reward_functions(self) -> list[RewardFunc]:
        return [_wrap_llm_judge_func(self.generation_wrapper, self.bench)]
