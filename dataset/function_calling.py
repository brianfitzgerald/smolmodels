from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional, List, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
import lightning.pytorch as pl
from model.utils import IGNORE_TOKEN_INDEX, create_and_clear_directory, FineTunerDataset
import os
import re
import torch
from torch import Tensor

ROLE_DICT = {
    "ASSISTANT": "assistant",
    "USER": "user",
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
        conversation_steps.append({"role": role, "content": message})

    return conversation_steps


class FunctionCallingDataModule(FineTunerDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        dataset = load_dataset("glaiveai/glaive-function-calling-v2")["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_cache/function_calling"
        create_and_clear_directory(cache_dir)
        cpu_count = min(len(os.sched_getaffinity(0)), 16)
        cpu_count = 1

        # Set format for PyTorch
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

        input_ids, attention_masks, labels = [], [], []
        for system, chat in zip(examples["system"], examples["chat"]):
            conversation_string = system + chat
            conversation_chatml = chatml_to_conversation(conversation_string)

            # N-1 messages are the prompt, then the last message is the expected output
            prompt_str: str = self.tokenizer.apply_chat_template(
                conversation_chatml[:-1], tokenize=False
            )  # type: ignore

            expected_output_str: str = self.tokenizer.apply_chat_template(
                [conversation_chatml[-1]], tokenize=False
            )  # type: ignore

            prompt_tokenized = self.tokenizer(
                prompt_str,
                return_tensors="pt",
            )

            expected_output_tokenized = self.tokenizer(
                expected_output_str,
                return_tensors="pt",
            )

            prompt_input_ids: Tensor = prompt_tokenized["input_ids"].squeeze() # type: ignore
            expected_output_input_ids: Tensor = expected_output_tokenized["input_ids"].squeeze() # type: ignore
            tokenized_prompt_len = prompt_input_ids.shape[0]
            ignore_labels = torch.full((tokenized_prompt_len,), IGNORE_TOKEN_INDEX)

            input_ids.append(prompt_input_ids)
            attention_masks.append(prompt_tokenized["attention_mask"])
            labels.append(torch.cat([ignore_labels, expected_output_input_ids], dim=0))  # type: ignore

        labels_t = torch.stack(labels)
        input_ids_t = torch.stack(input_ids)
        attention_masks_t = torch.stack(attention_masks)
        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_masks_t,
            "labels": labels_t,
        }
