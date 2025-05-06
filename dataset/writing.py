import asyncio
from collections import defaultdict
from functools import partial

from loguru import logger
from synthetic_data.generation import (
    GenerationWrapper,
    get_generation_wrapper,
)
from synthetic_data.creative_writing_bench.bench import CreativeWritingBench
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
        "prompt": example["instruction"],
    }


T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        # Check for a running loop in the *current* (caller) thread first.
        # This determines if we need to use the threading approach.
        needs_thread = True
    except RuntimeError:
        needs_thread = False

    if needs_thread:
        # If called from a thread with a running loop, execute in a separate thread
        # to avoid blocking the caller's loop and prevent deadlocks.
        result_container = []
        sync_event = threading.Event()  # Use an event for better synchronization

        def _run_in_thread():
            try:
                # Get or create an event loop *for this new thread*
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.info("No loop in thread, creating a new one.")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                logger.info(
                    f"Running coroutine in background thread {threading.get_ident()} using loop {loop}"
                )
                result = loop.run_until_complete(coro)
                result_container.append(result)
            except Exception as e:
                # Store exception to re-raise in the main thread
                logger.error(f"Exception in coroutine thread: {e}", exc_info=True)
                result_container.append(e)
            finally:
                # Signal that the thread has finished.
                # We are *not* explicitly closing the loop here anymore.
                # Let it be managed by the thread/asyncio policy.
                logger.info(
                    f"Background thread {threading.get_ident()} finished execution."
                )
                sync_event.set()

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        sync_event.wait()  # Wait for the thread to signal completion

        if not result_container:
            raise RuntimeError(
                "Coroutine thread finished without producing a result or exception."
            )

        result = result_container[0]
        if isinstance(result, Exception):
            raise result  # Re-raise exception captured in the thread
        return result

    else:
        # No event loop running in the caller thread; use asyncio.run for clean execution.
        logger.info("No running loop found in caller thread, using asyncio.run()")
        return asyncio.run(coro)


MAX_SCORE = 20


def llm_judge_func(
    prompts: list[str],
    completions: list[str],
    generation_wrapper: GenerationWrapper | None = None,
    bench: CreativeWritingBench | None = None,
    **kwargs,
) -> list[float]:
    async def run_score():
        assert generation_wrapper is not None
        assert bench is not None
        out = await score_writing(completions, prompts, bench, generation_wrapper)
        return out

    scores = run_sync(run_score())

    scores_per_category = defaultdict(list)
    score_vals = []
    for i, sample_scores in enumerate(scores):
        total = 0.0
        if len(sample_scores) == 0:
            logger.error(
                f"s not same length as completions: {len(sample_scores)} != {len(completions)}"
            )
            return [0.0] * len(completions)
        for k, v in sample_scores.items():
            v = max(min(v, MAX_SCORE), 0.0)
            total += v / MAX_SCORE
            scores_per_category[k].append(v)

        avg_score = round(total / len(sample_scores), 2)
        score_vals.append(avg_score)

    if len(score_vals) != len(completions):
        logger.error(
            f"score_vals not same length as completions: {len(score_vals)} != {len(completions)}"
        )
        score_vals = [0.0] * len(completions)
    logger.info(f"Sample scores - totals: {score_vals}")
    for k, v in scores_per_category.items():
        logger.info(f"\t{k}: {v}")
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


def antislop_func(
    prompts: list[dict[str, str]],
    completions: list[dict[str, str]],
    bench: CreativeWritingBench,
    **kwargs,
) -> list[float]:
    slop_scores = [bench.calculate_slop_index(completion) for completion in completions]
    slop_scores = [max(100 - score, 0) for score in slop_scores]
    logger.info(f"Slop scores - totals: {slop_scores}")
    return slop_scores


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
        self.generation_wrapper = get_generation_wrapper(self.config.judge_model)
        self.bench = CreativeWritingBench(self.config.run_mode)

    def reward_functions(self) -> list[RewardFunc]:
        antislop_fn = partial(antislop_func, bench=self.bench)
        antislop_fn.__name__ = "antislop_func"  # type: ignore
        return [
            create_llm_judge_func(self.generation_wrapper, self.bench),
            logger_reward,
            antislop_fn,
        ]
