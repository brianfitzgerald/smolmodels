from dataclasses import dataclass
from enum import IntEnum
import torch
import os
import platform
import requests
import torch.nn.functional as F
from transformers import AutoTokenizer
from pprint import pprint


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


def get_model_output(outputs, batch, device):
    logits = outputs["logits"]
    batch_size, seq_length, num_classes = logits.shape
    logits = logits.view(batch_size * seq_length, num_classes)
    input_ids = batch["input_ids"].view(-1).to(device)
    return logits, input_ids


def get_perplexity(logits: torch.Tensor, input_ids: torch.Tensor):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perp = torch.exp(loss)
    return perp.item()


def get_text_sample(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor = None,
):
    print("eval shapes", logits.shape, input_ids.shape)
    probs = torch.softmax(logits, dim=-1)
    token_index = torch.argmax(probs, dim=1)
    completions = tokenizer.decode(token_index)

    log_dict = {"completions": completions}
    if input_ids is not None:
        prompts = tokenizer.decode(input_ids)
        log_dict["prompts"] = prompts
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
