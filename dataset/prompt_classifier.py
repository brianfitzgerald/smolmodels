from transformers.tokenization_utils import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset
from typing import List
import torch
from torch import Tensor

from dataset.utils import FineTunerDataset
from synthetic_data.utils import ensure_directory, SAFERPROMPT_LABELS, ANNOTATED_LABELS
from torch.utils.data import DataLoader, WeightedRandomSampler

SAFERPROMPT_IDS_TO_LABELS = {
    SAFERPROMPT_LABELS[label]: label for label in SAFERPROMPT_LABELS
}


I2P_LABELS = {
    "hate": 0,
    "harassment": 1,
    "violence": 2,
    "self-harm": 3,
    "sexual-content": 4,
    "shocking-images": 5,
    "illegal-activity": 6,
}
I2P_IDS_TO_LABELS = {I2P_LABELS[label]: label for label in I2P_LABELS}


class ClipdropSyntheticClassesDataModule(FineTunerDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.prompt_column = "prompt"
        self.label_column = "class_label"
        self.cache_dir = "dataset_caches/clipdrop"
        self.data_file_path = "data_files/classified_prompts_40k_synthetic.csv"

    def setup(self, stage: str):
        print(f"Loading dataset for stage {stage}")
        dataset_csv = pd.read_csv(self.data_file_path)
        self.dataset = Dataset.from_pandas(dataset_csv)
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
            SAFERPROMPT_LABELS[label] for label in examples[self.label_column]
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
        self.label_column = "taskus_label"
        self.cache_dir = "dataset_caches/clipdrop_binary"
        self.data_file_path = "data_files/clipdrop_prompts_benchmark_annotated.csv"

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.filter(
            lambda x: x["prompt"] is not None and x["taskus_label"] is not None
        )
        return dataset

    def prepare_sample(self, examples: dict):

        inputs: List[str] = examples[self.prompt_column]

        labels: List[int] = [
            ANNOTATED_LABELS[label] for label in examples[self.label_column]
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

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)  # type: ignore
        return sampler

    def train_dataloader(self):
        sampler = self.get_sampler(self.train_dataset["labels"])  # type: ignore
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.cpu_count, sampler=sampler)  # type: ignore

    def val_dataloader(self):
        sampler = self.get_sampler(self.val_dataset["labels"])  # type: ignore
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_count, sampler=sampler)  # type: ignore
