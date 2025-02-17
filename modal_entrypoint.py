from pathlib import Path
from modal import Image, App, Secret
import os
from loguru import logger
import modal

from trl_wrapper.trainer_wrapper import CONFIGS, TrainerWrapper, MODELS_FOLDER
from generate import main as generate_main

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

weights_volume = modal.Volume.from_name("model-weights", create_if_missing=True)

model_volume_path = Path(MODELS_FOLDER)

MODAL_IMAGE = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": model_volume_path.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_commands(
        [
            "uv sync --frozen --group torch",
            "uv sync --group torch --group training --no-build-isolation",
        ]
    )
    .run_commands("pip install huggingface_hub[hf_transfer] hf_transfer")
    .add_local_dir("dataset_files", "/dataset_files", copy=True)
    .add_local_dir("chat_templates", "/chat_templates", copy=True)
    .add_local_file(".env", "/.env", copy=True)
    .add_local_python_source(
        "dataset", "evaluation", "generate", "model", "synthetic_data", "trl_wrapper"
    )
)

APP_NAME = "smolmodels"

app = App(
    APP_NAME,
    secrets=[
        Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
    ],
)


def _format_timeout(seconds: int = 0, minutes: int = 0, hours: int = 0):
    return seconds + minutes * 60 + hours * 60 * 60


@app.function(
    image=MODAL_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={model_volume_path.as_posix(): weights_volume},
    timeout=_format_timeout(hours=5),
)
def training(config: str = "playwright"):
    assert config in CONFIGS, f"Unknown config: {config}"

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, True)
    wrapper.init_model()
    wrapper.init_data_module(
        # dataset_root_path=model_volume_path.as_posix()
    )
    wrapper.init_trainer(config)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()


@app.function(
    image=MODAL_IMAGE,
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={model_volume_path.as_posix(): weights_volume},
    timeout=_format_timeout(hours=12),
)
def generation(task: str = "gutenberg_summarize"):
    generate_main(
        task_name=task,
        dataset_root_path=os.path.join(model_volume_path.as_posix(), "dataset_files"),
    )
