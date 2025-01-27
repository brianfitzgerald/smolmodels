from modal import Image, App, Secret
import os
from loguru import logger

from trl_wrapper.trainer_wrapper import CONFIGS, TrainerWrapper

MODAL_IMAGE = (
    Image.debian_slim()
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/uv.lock", copy=True)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands(
        [
            "uv sync --frozen --compile-bytecode --group torch",
            "uv sync --group torch --group training --no-build-isolation",
            "uv build",
        ]
    )
)

APP_NAME = "smolmodels"

app = App(
    APP_NAME,
    secrets=[
        Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
    ],
)


@app.function(image=MODAL_IMAGE, gpu="l40s")
def main(config: str = "playwright"):
    assert config in CONFIGS, f"Unknown config: {config}"

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, True)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer(config)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()
