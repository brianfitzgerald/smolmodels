from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import (
    DatasetConfig,
    SmDataset,
)
from synthetic_data.utils import ldictl
import numpy as np
from torch import Tensor

MASK_IDX = -100


class PlaywrightSummaryToScript(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)
        self.prompt_tag = "<summary>"
        self.response_tag = "<scene>"
        self.prompt_tag_token_ids = self.tokenizer(
            self.prompt_tag, add_special_tokens=False
        )["input_ids"]
        self.response_tag_token_ids = self.tokenizer(
            self.response_tag, add_special_tokens=False
        )["input_ids"]

    def process_samples_batch(self, examples: dict):
        """
        Tokenize the response, then find the tokens for the tags, and mask all before the response token
        """
        out = []
        for summary, scene in zip(examples["summary"], examples["scene"]):
            tokenized = self.tokenizer(
                self.prompt_tag + summary + self.response_tag + scene,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids: Tensor = tokenized["input_ids"]  # type: ignore
            input_ids = input_ids.squeeze()
            labels = input_ids.clone()
            tag_id = self.response_tag_token_ids[0]  # type: ignore
            response_tag_idx = int((input_ids == tag_id).nonzero()[0])
            labels[:response_tag_idx] = MASK_IDX
            tokenized["labels"] = labels
            out_batch = {
                "input_ids": input_ids,
                "labels": labels,
            }
            out.append(out_batch)
        return ldictl(out)

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns(
            ["name", "scene", "summary"]
        )
        self.val_dataset = self.val_dataset.remove_columns(["name", "scene", "summary"])
