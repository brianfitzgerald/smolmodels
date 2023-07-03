from dataclasses import dataclass
from enum import IntEnum
import torch
import os
import platform
import requests
import torch.nn.functional as F
from transformers import AutoTokenizer


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


class DatasetChoice(IntEnum):
    CRD = 1
    ROLEPLAY_INSTRUCT = 2


@dataclass
class TrainingArgs:
    ds_choice = DatasetChoice.CRD
    num_epochs = 24
    batch_size = 8
    save_interval = 2
    eval_interval = 1
    use_wandb = True
    push_model = False
    model_name = "smolmodels-finetune-33m-dialogue"
    use_peft = False


def get_perplexity(logits: torch.Tensor, input_ids: torch.Tensor):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perp = torch.exp(loss)
    return perp.item()


def get_text_sample(
    logits: torch.Tensor, input_ids: torch.Tensor, tokenizer: AutoTokenizer
):
    decoded_input = tokenizer.batch_decode(input_ids)
    sample_batchsize, seq_length, _ = logits.size()
    decoded_texts = []

    for i in range(sample_batchsize):
        decoded_tokens = []
        for j in range(seq_length):
            token_id = torch.argmax(logits[i, j]).item()
            token = tokenizer.decode([token_id])
            decoded_tokens.append(token)

        decoded_text = tokenizer.convert_tokens_to_string(decoded_tokens)
        decoded_texts.append(decoded_text)

    return decoded_input, decoded_texts


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
