from pathlib import Path
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import Tensor


def generate_full_prompt(instruction: str, prompt: str) -> str:
    """
    Generates a prompt for a given example.
    """

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n\n### Response:"
    )


class PromptUpsampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_token_length: int,
        sequence_to_sequence: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.train_dataset = None
        self.val_dataset = None
        self.sequence_to_sequence = sequence_to_sequence
        self.mask_inputs = True

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        # Load dataset and split
        dataset = load_dataset("roborovski/upsampled-prompts-parti")["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_cache"
        Path(cache_dir).mkdir(exist_ok=True)

        # Tokenization
        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{cache_dir}/training",
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{cache_dir}/validation",
        )

        columns = [
            "input_ids",
            "attention_mask",
            "labels",
        ]

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def prepare_sample(self, examples: dict):
        processed_batch = {}

        if self.sequence_to_sequence:
            inputs = ["Expand the following prompt to add more detail: " + doc for doc in examples["Prompt"]]

            inputs_tokenized = self.tokenizer(
                inputs,
                max_length=self.max_token_length,
                truncation=True,
                padding="max_length",
            )

            labels_tokenized = self.tokenizer(
                text_target=examples["Upsampled"],
                max_length=self.max_token_length,
                truncation=True,
                padding="max_length",
            )

            return {
                "input_ids": inputs_tokenized["input_ids"],
                "attention_mask": inputs_tokenized["attention_mask"],
                "labels": labels_tokenized["input_ids"],
            }

        else:
            prompts, outputs = examples["Prompt"], examples["Upsampled"]

            instruction = "Expand the following description:"
            full_prompts = [generate_full_prompt(instruction, prompt) for prompt in prompts]

            full_prompts_and_outputs = [
                f"{prompt}\n\n{upsampled}"
                for prompt, upsampled in zip(full_prompts, outputs)
            ]

            full_prompts_encoded = self.tokenizer(
                full_prompts,
                max_length=self.max_token_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            full_prompts_and_outputs_encoded = self.tokenizer(
                full_prompts_and_outputs,
                max_length=self.max_token_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            labels: Tensor = full_prompts_and_outputs_encoded["input_ids"] # type: ignore
            input_ids: Tensor = full_prompts_and_outputs_encoded["input_ids"].clone() # type: ignore
            if self.mask_inputs:
                labels.masked_fill_(full_prompts_encoded["attention_mask"] == 1, -100) # type: ignore
            return {
                "input_ids": input_ids,
                "attention_mask": full_prompts_and_outputs_encoded["attention_mask"],
                "labels": labels,
            }


        return processed_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=35)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=35)  # type: ignore
