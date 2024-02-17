from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import Tensor
import lightning.pytorch as pl

from model.utils import TASK_PREFIX

class LlavaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        dataset = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K")["train"].train_test_split(test_size=0.01) # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_cache_llava"
        Path(cache_dir).mkdir(exist_ok=True)
        
        # Set format for PyTorch
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

    def prepare_sample(self, examples: dict):
        processed_batch = {}

        inputs = [
            TASK_PREFIX + doc
            for doc in examples["Prompt"]
        ]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            text_target=examples["Upsampled"],
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "decoder_attention_mask": labels_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"],
        }
        return processed_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=35)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=35)  # type: ignore
