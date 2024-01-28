import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from typing import Optional


class PromptUpsampleDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size=8, max_token_length=512):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.prefix = "Expand: "

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")
        # Load dataset and split
        dataset = load_dataset("roborovski/upsampled-prompts-parti")["train"].train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        # Tokenization
        self.train_dataset = self.train_dataset.map(self.tokenize_fn, batched=True)
        self.val_dataset = self.val_dataset.map(self.tokenize_fn, batched=True)

        columns = [
            "input_ids",
            "attention_mask",
            "label_input_ids",
            "label_attention_mask",
        ]
        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def tokenize_fn(self, examples):
        inputs = [self.prefix + doc for doc in examples["Prompt"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
        )

        labels = self.tokenizer(
            text_target=examples["Upsampled"],
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["label_input_ids"] = labels["input_ids"]
        model_inputs["label_attention_mask"] = labels["attention_mask"]
        return model_inputs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)  # type: ignore
