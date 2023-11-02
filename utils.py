from dataclasses import dataclass
from enum import IntEnum
import torch
import os
import platform
import requests
import torch.nn.functional as F
from transformers import AutoTokenizer
from pprint import pprint
from icecream import ic
from pathlib import Path
import sys
import math
import pickle
import sys
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, ContextManager, Dict, List, Mapping, Optional, TypeVar, Union

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from torch.serialization import normalize_storage_type


def get_available_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def should_use_wandb():
    if os.environ.get("NO_WANDB", False):
        return False
    return os.environ.get("USER") == "ubuntu" and platform.system().lower() == "linux"


# https://github.com/teknium1/GPTeacher/blob/main/Roleplay/roleplay-simple-deduped-roleplay-instruct.json


def download_if_not_present(file_path: str, url: str):
    # Define the file path and URL

    # Check if the file exists
    if os.path.exists(file_path):
        print("File already exists.")
    else:
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download the file.")


class Task(IntEnum):
    TINY_STORIES = 1
    STATE_CHANGES = 2
    CRD = 3
    ROLEPLAY_INSTRUCT = 4


def get_model_output_for_loss(outputs, batch, device):
    breakpoint()
    logits = outputs["logits"]
    batch_size, seq_length, num_classes = logits.shape
    logits = logits.view(batch_size * seq_length, num_classes)
    input_ids = batch["input_ids"].view(-1).to(device)
    return logits, input_ids


def get_perplexity(logits: torch.Tensor, input_ids: torch.Tensor):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perp = torch.exp(loss)
    return perp.item()


def get_completion_samples(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
):

    probs = F.softmax(logits, dim=1)
    token_index = torch.argmax(probs, dim=2)
    # iterate over batch
    completions = []
    for i in range(token_index.shape[0]):
        sample_completion = tokenizer.decode(token_index[i]) # type: ignore
        sample_prompt = tokenizer.decode(input_ids[i]) # type: ignore
        print('prompt:\n', sample_prompt, '\ncompletion:\n', sample_completion)
        completions.append((sample_prompt, sample_completion))

    log_dict = {"completions": completions}
    return log_dict


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    files = {
        "lit_model.pth": (checkpoint_dir / "lit_model.pth").is_file(),
        "lit_config.json": (checkpoint_dir / "lit_config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or (
            checkpoint_dir / "tokenizer.model"
        ).is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials\n"
        f"{extra}\nSee all download options by running:\n python scripts/download.py"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)

def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)

def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)
