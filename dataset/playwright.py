from typing import List
from model.utils import (
    SmDataset,
)
from synthetic_data.utils import dictl

MASK_IDX = -100


class PlaywrightSummaryToScript(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def process_samples_batch(self, examples: dict):
        examples_list = dictl(examples)
        out = []
        return examples

    def _tokenize(self, inputs: List[str], labels: List[str]) -> dict:
        """
        Basic tokenizing function. Inputs are samples from the dataset, and labels are the target values.
        """
        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            labels,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"],
        }
