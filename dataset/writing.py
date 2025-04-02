from trl_wrapper.wrapper_config import SmDataset
from datasets import Dataset
from typing import Optional
import pandas as pd


def _sum_scores(scores: dict[str, float | None] | None) -> float:
    if scores is None:
        return 0.0
    total = 0.0
    for v in scores.values():
        if v is None:
            continue
        total += v
    return total


def _pick_completions(group):
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

        dataset_pd["score_total"] = dataset_pd["scores"].apply(_sum_scores)

        dataset_pd = dataset_pd.groupby("instruction_id").filter(
            lambda group: group["score_total"].max() != group["score_total"].min()
        )
        dataset_pd = (
            dataset_pd.groupby("instruction_id").apply(_pick_completions).reset_index()
        )
        dataset = Dataset.from_pandas(dataset_pd).train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]


class WritingGRPODataModule(SmDataset):
    def setup(self, stage: Optional[str] = None):
        dataset_pd = pd.read_parquet("../dataset_files/backtranslate_best_of_n.parquet")  # type: ignore
        if self.config.max_samples is not None:
            dataset_pd = dataset_pd.head(self.config.max_samples)
        dataset_pd["instruction_id"] = (
            dataset_pd["instruction"].astype("category").cat.codes
        )

        dataset_pd["score_total"] = dataset_pd["scores"].apply(_sum_scores)

        dataset_pd = dataset_pd.groupby("instruction_id").filter(
            lambda group: group["score_total"].max() != group["score_total"].min()
        )
        result = dataset_pd.groupby("instruction_id").all()
        dataset = Dataset.from_pandas(result).train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
