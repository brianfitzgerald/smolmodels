print("Loading torch")
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TypedDict

print("Loading lightning")
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks

print("Loading HF")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset


class PromptUpsampleDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size=8, max_token_length=512):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.prefix = "Expand: "

    def setup(self, stage=None):
        # Load dataset and split
        dataset = load_dataset("roborovski/upsampled-prompts-parti")["train"].train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        # Tokenization
        self.train_dataset = self.train_dataset.map(self.tokenize_fn, batched=True)
        self.val_dataset = self.val_dataset.map(self.tokenize_fn, batched=True)

        # Set format for PyTorch
        self.train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        self.val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    def tokenize_fn(self, examples):
        model_inputs = [self.prefix + doc for doc in examples["Prompt"]]
        model_inputs = self.tokenizer(
            model_inputs, max_length=self.max_token_length, truncation=True
        )

        labels = self.tokenizer(
            text_target=examples["Upsampled"],
            max_length=self.max_token_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)  # type: ignore