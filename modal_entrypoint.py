import os
from loguru import logger
import modal
import modal.gpu
import subprocess
import signal
import sys


from scripts.modal_definitions import (
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
    gpu="A100:2",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=6),
)
def training(config: str, wandb: bool = True):
    assert config, "Config must be provided with --config"
    assert config in CONFIGS, f"Unknown config: {config}"
    cfg = CONFIGS[config]
    vllm_process = None
    if cfg.tuning_mode == "grpo":
        cmd = (
            f"uv run trl vllm-serve --model {cfg.model_id_or_path} --max_model_len 4096"
        )
        cmd_list = cmd.split()
        logger.info(f"Starting vLLM server, cmd: {cmd_list}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        vllm_process = subprocess.Popen(
            cmd_list,
            stdout=sys.stdout,
            stderr=sys.stdout,
            env=env,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def cleanup():
        if vllm_process:
            vllm_process.terminate()
            vllm_process.wait()

    signal.signal(signal.SIGTERM, lambda signo, frame: cleanup())
    signal.signal(signal.SIGINT, lambda signo, frame: cleanup())

    try:
        wrapper = TrainerWrapper(cfg, use_wandb=wandb, modal=True)
        wrapper.init_model()
        wrapper.init_data_module(dataset_root_path=DATASET_VOLUME_PATH)
        wrapper.init_trainer(config)
        logger.info(f"Starting training, config: {config}")
        wrapper.train()
    finally:
        cleanup()


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
