from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional, List, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
import lightning.pytorch as pl
from model.utils import IGNORE_TOKEN_INDEX, create_and_clear_directory
import os
import re
import torch
from torch import Tensor

ROLE_DICT = {
    "ASSISTANT": "assistant",
    "USER": "human",
    "SYSTEM": "system",
}


def chatml_to_conversation(conversation: str) -> List[Dict]:
    """
    Convert a string in ChatML format to a list of conversation steps.
    """
    message_regex = r"(SYSTEM|USER|ASSISTANT):"

    conversation = conversation.replace("<|endoftext|>", "").replace("\n", " ")

    conversation_steps = []
    user_and_assistant_messages = re.split(message_regex, conversation)
    messages = [
        message.strip() for message in user_and_assistant_messages if message.strip()
    ]
    for i in range(0, len(messages), 2):
        if messages[i] not in ROLE_DICT:
            print(f"Unknown role: {messages[i]}")
            continue
        role = ROLE_DICT[messages[i]]
        if i + 1 >= len(messages):
            print(f"Missing message for role: {role}")
            continue
        message = messages[i + 1]
        conversation_steps.append({"from": role, "value": message})

    return conversation_steps


class FunctionCallingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        dataset = load_dataset("glaiveai/glaive-function-calling-v2")["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_cache_function_calling"
        create_and_clear_directory(cache_dir)
        cpu_count = os.cpu_count() or 16

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/training",
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/training",
        )

    def prepare_sample(self, examples: dict):

        conversation_string = examples["system"] + " " + examples["chat"]
        conversation_chatml = chatml_to_conversation(conversation_string)

        # N-1 messages are the prompt, then the last message is the expected output
        prompt, expected_output = conversation_chatml[:-1], conversation_chatml[-1]

        tokenized_prompt = self.tokenizer.apply_chat_template(
            prompt,
            max_length=self.max_token_length,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_expected_output = self.tokenizer.apply_chat_template(
            [expected_output],
            max_length=self.max_token_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids: Tensor = tokenized_prompt["input_ids"]  # type: ignore
        attention_mask: Tensor = tokenized_expected_output["attention_mask"]  # type: ignore
        tokenized_prompt_len = len(tokenized_prompt["input_ids"])  # type: ignore
        expected_input_ids: Tensor = tokenized_expected_output["input_ids"]  # type: ignore

        labels = torch.full((tokenized_prompt_len,), IGNORE_TOKEN_INDEX)
        labels = torch.cat([labels, expected_input_ids], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=35)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=35)  # type: ignore
