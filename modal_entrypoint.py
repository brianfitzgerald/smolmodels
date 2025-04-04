import os
from loguru import logger
import modal
import modal.gpu

from scripts.modal_definitons import (
    SMOLMODELS_IMAGE,
    format_timeout,
    app,
    MODELS_VOLUME_PATH,
    MODEL_WEIGHTS_VOLUME,
)
from trl_wrapper.trainer_wrapper import CONFIGS, TrainerWrapper
from generate import main as generate_main

DATASET_VOLUME_PATH = os.path.join(MODELS_VOLUME_PATH.as_posix(), "dataset_files")


@app.function(
    image=SMOLMODELS_IMAGE,
    gpu=modal.gpu.A100(size="80GB"),
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=5),
)
def training(config: str = "grpo_connections"):
    assert config in CONFIGS, f"Unknown config: {config}"

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, True)
    wrapper.init_model()
    wrapper.init_data_module(dataset_root_path=DATASET_VOLUME_PATH)
    wrapper.init_trainer(config)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()


@app.function(
    image=SMOLMODELS_IMAGE,
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=12),
)
def generation():
    logger.info(
        f"Dataset root path: {DATASET_VOLUME_PATH} contents: {os.listdir(DATASET_VOLUME_PATH)}"
    )
    generate_main(task_name="backtranslate_best_of_n", run_mode="modal")
