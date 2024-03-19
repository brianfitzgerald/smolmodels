from transformers.tokenization_utils import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset
from typing import List
import torch


from dataset.utils import FineTunerDataset
from synthetic_data.utils import SAFE_PROMPT_LABEL_IDS, ensure_directory


class PromptClassifierDataModule(FineTunerDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.prompt_column = "prompt"
        self.label_column = "class_label"
        self.cache_dir = "dataset_caches/saferprompt_classifier"

    def setup(self, stage: str):
        print(f"Loading dataset for stage {stage}")
        dataset_csv = pd.read_csv("data_files/classified_prompts_40k.csv")
        self.dataset = Dataset.from_pandas(dataset_csv).train_test_split(test_size=0.01)
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

    def prepare_sample(self, examples: dict):

        inputs: List[str] = examples[self.prompt_column]
        labels: List[int] = [
            SAFE_PROMPT_LABEL_IDS[label] for label in examples[self.label_column]
        ]
        # convert to binary classification
        # labels = [1 if label != 0 else 0 for label in labels]
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
