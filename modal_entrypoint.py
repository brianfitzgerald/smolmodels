import os
from loguru import logger
import modal

from scripts.modal_definitons import (
    SMOLMODELS_IMAGE,
    format_timeout,
    app,
    MODELS_VOLUME_PATH,
    MODEL_WEIGHTS_VOLUME,
)
from trl_wrapper.trainer_wrapper import CONFIGS, TrainerWrapper
from generate import main as generate_main

DATASET_ROOT_PATH = os.path.join(MODELS_VOLUME_PATH.as_posix(), "dataset_files")


@app.function(
    image=SMOLMODELS_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=5),
)
def training(config: str = "grpo_math"):
    assert config in CONFIGS, f"Unknown config: {config}"

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, True)
    wrapper.init_model()
    wrapper.init_data_module(dataset_root_path=DATASET_ROOT_PATH)
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
        f"Dataset root path: {DATASET_ROOT_PATH} contents: {os.listdir(DATASET_ROOT_PATH)}"
    )
    generate_main(
        task_name="gutenberg_backtranslation_from_txt",
        dataset_root_path=DATASET_ROOT_PATH,
        model="gemini-2.0-flash",
    )

    # generate_main(
    #     environment_name="twenty_questions",
    #     dataset_root_path=dataset_root_path,
    #     model="o3-mini",
    #     n_epochs=100,
    #     batch_size=32,
    #     save_every_n_batches=1,
    # )
