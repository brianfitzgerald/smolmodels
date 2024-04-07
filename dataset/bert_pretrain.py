from datasets import load_dataset, concatenate_datasets, Dataset
from typing import Optional, List, Dict
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from model.utils import IGNORE_TOKEN_INDEX, ensure_directory, FineTunerDataset
import os
import re
import torch
from torch import Tensor
import torch.nn.functional as F
from unidecode import unidecode

from synthetic_data.conversion import chatml_to_conversation
import lightning.pytorch as pl


def clean_bookcorpus_text(text: str) -> str:
    s = unidecode(text)
    s = s.lower()
    s = re.sub(
        "[ \t]+", " ", s
    )  # Replace tabs and sequences of spaces with a single space
    s = s.replace("\n", "\\n")
    return s.strip()


class BertPretrainDataset(pl.LightningDataModule):
    def __init__(
        self,
        max_token_length: int,
    ):

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.max_token_length = max_token_length
        self.cpu_count = min(len(os.sched_getaffinity(0)), 16)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-uncased"
        )

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        bc: Dataset = load_dataset("bookcorpus", split="train")  # type: ignore
        wp: Dataset = load_dataset("wikipedia", "20220301.en", split="train[0:5000000]")  # type: ignore

        full_dataset = concatenate_datasets([bc, wp])
        full_dataset = full_dataset.train_test_split(test_size=0.01)  # type: ignore

        self.train_dataset = full_dataset["train"]
        self.val_dataset = full_dataset["test"]

        cache_dir = "dataset_caches/bert_pretrain"

        ensure_directory(cache_dir, clear=False)
        # cpu_count = min(len(os.sched_getaffinity(0)), 16)  # type: ignore
        cpu_count = 1

        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def prepare_sample(self, examples: dict):
        """
        Parse chatml string to conversation steps, convert to prompt and output, and then tokenize
        Tokenizing is split from applying the chat template so we can output the attention mask
        """

        inputs = [clean_bookcorpus_text(doc) for doc in examples["text"]]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
        }
