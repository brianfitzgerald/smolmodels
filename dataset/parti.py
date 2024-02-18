from pathlib import Path
from datasets import load_dataset
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import Tensor
import lightning.pytorch as pl

from model.utils import PROMPT_EXPANSION_TASK_PREFIX, ensure_directory, FineTunerDataset


def generate_full_prompt(instruction: str, prompt: str) -> str:
    """
    Generates a prompt for a given example.
    """

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n\n### Response:"
    )


class PromptUpsampleDataModule(FineTunerDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.sequence_to_sequence = True
        self.load_local = False
        self.mask_inputs = True

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        # Load dataset and split
        if self.load_local:
            dataset = load_dataset("parquet", data_files={"train": "parti_prompts.parquet"})["train"].train_test_split(test_size=0.01)  # type: ignore
        else:

            dataset = load_dataset("roborovski/upsampled-prompts-parti")["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_caches/parti"
        ensure_directory(cache_dir)

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
            "decoder_attention_mask",
            "labels",
        ]

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def prepare_sample(self, examples: dict):
        processed_batch = {}

        inputs = [
            PROMPT_EXPANSION_TASK_PREFIX + doc
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
