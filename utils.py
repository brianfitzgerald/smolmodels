import torch
import os
import platform
import requests

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
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download the file.")
