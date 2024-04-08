from datasets import load_dataset, concatenate_datasets, Dataset
from typing import Optional
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from model.utils import ensure_directory, SmDataset
import os
import re
from unidecode import unidecode
from transformers.tokenization_utils import PreTrainedTokenizer

import lightning.pytorch as pl


def clean_bookcorpus_text(text: str) -> str:
    s = unidecode(text)
    s = s.lower()
    s = re.sub(
        "[ \t]+", " ", s
    )  # Replace tabs and sequences of spaces with a single space
    s = s.replace("\n", "\\n")
    return s.strip()


class BertPretrainDataset(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.max_token_length = max_token_length
        # self.cpu_count = min(len(os.sched_getaffinity(0)), 16)
        self.cpu_count = 1

    def prepare_data(self) -> None:
        bc: Dataset = load_dataset("saibo/bookcorpus_deduplicated_small", split="train")  # type: ignore
        # wp: Dataset = load_dataset("wikipedia", "20220301.en", split="train[0:100000]")  # type: ignore

        self.full_dataset = concatenate_datasets([bc]).train_test_split(test_size=0.01)

        self.train_dataset = self.full_dataset["train"]
        self.val_dataset = self.full_dataset["test"]
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/bert_pretrain"

        ensure_directory(cache_dir, clear=False)
        # cpu_count = min(len(os.sched_getaffinity(0)), 16)  # type: ignore
        cpu_count = 1

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def prepare_sample(self, examples: dict):

        inputs = [clean_bookcorpus_text(doc) for doc in examples["text"]]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "token_type_ids": inputs_tokenized["token_type_ids"],
            "overflow_to_sample_mapping": inputs_tokenized[
                "overflow_to_sample_mapping"
            ],
        }
