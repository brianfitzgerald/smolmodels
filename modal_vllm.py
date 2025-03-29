import os
from typing import Optional
import modal
from loguru import logger
from scripts.modal_definitons import MODEL_WEIGHTS_VOLUME
import subprocess


# https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_inference.py


def get_checkpoint_dir(
    base_run_dir: str,
    model_id: Optional[str] = None,
    run: Optional[str] = None,
    steps: Optional[int] = None,
) -> str:
    checkpoint_dir = model_id
    logger.info(f"model id: {model_id}, run: {run}, steps: {steps}")
    if run:
        run_directory = os.path.join(base_run_dir, run)
        logger.info(f"run_directory: {run_directory}")
        if not os.path.exists(run_directory):
            raise ValueError(f"Run directory {run_directory} not found")
        checkpoints = os.listdir(run_directory)
        logger.info(f"checkpoints: {checkpoints}")
        checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
        sorted_checkpoints = list(
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        )
        sorted_checkpoints = [int(x.split("-")[-1]) for x in sorted_checkpoints]
        if steps in sorted_checkpoints:
            checkpoint_dir = f"{run_directory}/checkpoint-{steps}"
        elif steps is not None and steps not in sorted_checkpoints:
            raise ValueError(f"Checkpoint {steps} not found in {run_directory}")
        else:
            latest_ckpt = sorted_checkpoints[-1]
            checkpoint_dir = f"{run_directory}/checkpoint-{latest_ckpt}"
    assert checkpoint_dir is not None
    return checkpoint_dir


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "loguru",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)


app = modal.App("vllm-server")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "super-secret-key"  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds

VLLM_PORT = 8000


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MODEL_NAME = "03-21-3-7-766288-llama-3.2-3b-instruct-txt_bt-txt-bt"


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    # how many requests can one replica handle? tune carefully!
    allow_concurrent_inputs=100,
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/model-weights": MODEL_WEIGHTS_VOLUME,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    model_checkpoint = get_checkpoint_dir(
        base_run_dir="/model-weights/runs",
        run=MODEL_NAME,
        steps=None,
    )

    logger.info(f"model checkpoint: {model_checkpoint}")

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_checkpoint,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
