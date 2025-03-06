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


@app.function(
    image=SMOLMODELS_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=5),
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
    image=SMOLMODELS_IMAGE,
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=12),
)
def generation(task_name: str):
    dataset_root_path = os.path.join(MODELS_VOLUME_PATH.as_posix(), "dataset_files")
    # generate_main(
    #     task_name=task_name,
    #     dataset_root_path=dataset_root_path,
    # )

    generate_main(
        environment_name="twenty_questions",
        dataset_root_path=dataset_root_path,
        model="deepseek-r1",
        n_epochs=20,
        batch_size=4,
        save_every_n_batches=1,
    )
