from trl_wrapper.wrapper_config import SmDataset
from datasets import Dataset
from typing import Optional
import pandas as pd


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
