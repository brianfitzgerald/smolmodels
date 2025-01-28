from pathlib import Path
from modal import Image, App, Secret
import os
from loguru import logger
import modal

from trl_wrapper.trainer_wrapper import CONFIGS, TrainerWrapper

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

volume = modal.Volume.from_name("model-weights", create_if_missing=True)
MODEL_DIR = Path("/models")


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
            "HF_HOME": MODEL_DIR.as_posix(),
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
    .add_local_file(
        "pippa_conversations.parquet", "/pippa_conversations.parquet", copy=True
    )
    .add_local_dir("chat_templates", "/chat_templates", copy=True)
)

APP_NAME = "smolmodels"

app = App(
    APP_NAME,
    secrets=[
        Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
    ],
)


@app.function(
    image=MODAL_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODEL_DIR.as_posix(): volume},
    timeout=60 * 5,
)
def main(config: str = "playwright"):
    assert config in CONFIGS, f"Unknown config: {config}"

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, True)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer(config)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()
