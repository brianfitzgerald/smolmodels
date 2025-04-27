from pathlib import Path
from modal import Image, App, Secret
import os
import modal

APP_NAME = "smolmodels"
MODELS_FOLDER = "/model-weights"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

MODEL_WEIGHTS_VOLUME = modal.Volume.from_name("model-weights", create_if_missing=True)

MODELS_VOLUME_PATH = Path(MODELS_FOLDER)

app = App(
    APP_NAME,
    secrets=[
        Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
    ],
)

SMOLMODELS_IMAGE = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": MODELS_VOLUME_PATH.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_commands(
        [
            "uv sync",
            "uv sync --group torch --group training --no-build-isolation",
        ]
    )
    .run_commands("uv pip install huggingface_hub[hf_transfer] hf_transfer --system")
    .add_local_python_source(
        "dataset",
        "evaluation",
        "generate",
        "model",
        "synthetic_data",
        "trl_wrapper",
        "gyms",
        "scripts",
    )
    .add_local_dir("dataset_files", "/dataset_files", copy=False)
    .add_local_dir("chat_templates", "/chat_templates", copy=False)
    .add_local_file(".env", "/.env", copy=False)
)


def format_timeout(seconds: int = 0, minutes: int = 0, hours: int = 0):
    return seconds + (minutes * 60) + (hours * 60 * 60)


VLLM_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": MODELS_VOLUME_PATH.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands(
        ["uv pip install vllm==0.7.1 fastapi[standard]==0.115.8 loguru fire --system"]
    )
    .run_commands("uv pip install huggingface_hub[hf_transfer] hf_transfer --system")
    .add_local_python_source("scripts")
)
