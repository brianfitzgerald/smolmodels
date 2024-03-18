from transformers.tokenization_utils import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset

from model.utils import (
    PROMPT_EXPANSION_TASK_PREFIX,
    ensure_directory,
    SAFETY_TASK_PREFIX,
)

from dataset.utils import FineTunerDataset


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

    def setup(self, stage: str):
        print(f"Loading dataset for stage {stage}")
        dataset_csv = pd.read_csv("data_files/classified_prompts_40k.csv")
        self.dataset = Dataset.from_pandas(dataset_csv)

    def prepare_sample(self, examples: dict):

        inputs = examples[self.label_column]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            text_target=examples[self.label_column],
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"],
        }


