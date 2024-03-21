from transformers.tokenization_utils import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset, concatenate_datasets
from typing import List
import torch
from torch import Tensor
import csv

from dataset.utils import FineTunerDataset
from synthetic_data.labels import SAFERPROMPT_LABELS, ANNOTATED_LABELS
from synthetic_data.utils import ensure_directory
from torch.utils.data import DataLoader, WeightedRandomSampler


class ClipdropSyntheticClassesDataModule(FineTunerDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.prompt_column = "prompt"
        self.cache_dir = "dataset_caches/clipdrop"
        self.data_file_paths = ["data_files/classified_prompts_40k_synthetic.csv"]
        self.rename_columns = {}

    def setup(self, stage: str):
        print(f"Loading dataset for stage {stage}")
        all_datasets = []
        for data_file in self.data_file_paths:
            if data_file.endswith(".csv"):
                dataset_csv = pd.read_csv(data_file, on_bad_lines='skip', quoting=csv.QUOTE_NONE, encoding='utf-8')
            elif data_file.endswith(".tsv"):
                dataset_csv = pd.read_csv(data_file, on_bad_lines='skip', encoding='utf-8', sep='\t')
            dataset_csv = dataset_csv.rename(columns=self.rename_columns)
            dataset = Dataset.from_pandas(dataset_csv)
            all_datasets.append(dataset)

        self.dataset = concatenate_datasets(all_datasets)
        self.dataset = self.filter_dataset(self.dataset).train_test_split(
            test_size=0.01, seed=42
        )
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["test"]

        ensure_directory(self.cache_dir, clear=False)

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.cpu_count,
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.cpu_count,
        )

        columns = [
            "input_ids",
            "attention_mask",
            "labels",
        ]

        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def prepare_sample(self, examples: dict):

        inputs: List[str] = examples[self.prompt_column]

        labels: List[int] = [
            SAFERPROMPT_LABELS[label] for label in examples["tasks_label"]
        ]

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        inputs_tokenized = self.tokenizer(
            inputs,
            add_special_tokens=True,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tensor,
        }


class ClipdropBinaryDataModule(ClipdropSyntheticClassesDataModule):
    def __init__(
        self, batch_size: int, tokenizer: PreTrainedTokenizer, max_token_length: int
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.cache_dir = "dataset_caches/clipdrop_binary"
        self.data_file_paths = ["data_files/clipdrop_prompts_40k.tsv", "data_files/clipdrop_prompts_160k_batch1.tsv"]
        self.rename_columns = {"taskus_label": "tasks_label"}
        ensure_directory(self.cache_dir, clear=False)
        self.oversample = True

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """
        Minority oversample, drop na, and convert borderline to unsafe
        """
        dataset_pd: pd.DataFrame = dataset.to_pandas() # type: ignore
        dataset_pd = dataset_pd[["prompt", "tasks_label"]]
        dataset_pd.loc[dataset_pd["tasks_label"] == "borderline", "tasks_label"] = "unsafe"
        dataset_pd = dataset_pd.dropna(subset=["prompt", "tasks_label"])
        label_counts = dataset_pd["tasks_label"].value_counts()
        if self.oversample:
            max_count = label_counts.max()
            balanced_df = pd.concat([dataset_pd[dataset_pd['tasks_label'] == label].sample(max_count, replace=True) for label in label_counts.index])
            balanced_ds = Dataset.from_pandas(balanced_df)
        else:
            min_label_count = label_counts.min()
            under_sampled_df = pd.DataFrame()
            for label in label_counts.index:
                label_subset = dataset_pd[dataset_pd['tasks_label'] == label]
                under_sampled_label_subset = label_subset.sample(n=min_label_count, random_state=1)  # random_state for reproducibility
                under_sampled_df = pd.concat([under_sampled_df, under_sampled_label_subset], axis=0)
            balanced_ds = Dataset.from_pandas(under_sampled_df)
        return balanced_ds

    def prepare_sample(self, examples: dict):

        inputs: List[str] = examples[self.prompt_column]

        labels: List[int] = [
            ANNOTATED_LABELS[label] for label in examples["tasks_label"]
        ]

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        inputs_tokenized = self.tokenizer(
            inputs,
            add_special_tokens=True,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tensor,
        }

    def get_sampler(self, labels: Tensor):

        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]

        print(f"Class counts: {class_counts.tolist()}")

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)  # type: ignore
        return sampler

class ClipdropSafetyFamousFiguresDataModule(ClipdropSyntheticClassesDataModule):
    """
    Datamodule with 2 classes - safe and famous_figures
    """
    def __init__(
        self, batch_size: int, tokenizer: PreTrainedTokenizer, max_token_length: int
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.cache_dir = "dataset_caches/clipdrop_multilabel"
        self.data_file_paths = ["data_files/clipdrop_prompts_famous_figures.csv"]
        ensure_directory(self.cache_dir, clear=False)

    def prepare_sample(self, examples: dict):

        annotated_labels = [
            1 if label == "safe" else 0 for label in examples["annotated_label"]
        ]
        class_labels = [
            1 if label == "positive" else 0 for label in examples["class_label"]
        ]

        labels_batch = [
            [annotated_labels[i], class_labels[i]] for i in range(len(annotated_labels))
        ]

        labels_batch_tensor = torch.tensor(labels_batch, dtype=torch.long)

        inputs_tokenized = self.tokenizer(
            examples[self.prompt_column],
            add_special_tokens=True,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_batch_tensor,
        }