import os
from typing import Optional
import modal
from loguru import logger
import modal.gpu
from scripts.modal_definitions import MODEL_WEIGHTS_VOLUME
import subprocess
from dotenv import load_dotenv

load_dotenv()

# https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_inference.py


def get_checkpoint_dir(
    base_run_dir: str,
    model_id: str,
    steps: Optional[int] = None,
) -> str:
    checkpoint_dir = model_id
    logger.info(f"model id: {model_id}, steps: {steps}")
    run_directory = os.path.join(base_run_dir, model_id)
    logger.info(f"run_directory: {run_directory}")
    if os.path.exists(run_directory):
        logger.info(f"Run directory {run_directory} found, assuming fine tuned model")
        checkpoints = os.listdir(run_directory)
        checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
        sorted_checkpoint_steps = list(
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        )
        sorted_checkpoint_steps = [
            int(x.split("-")[-1]) for x in sorted_checkpoint_steps
        ]
        logger.info(f"sorted_checkpoint_steps: {sorted_checkpoint_steps}")
        if steps in sorted_checkpoint_steps:
            checkpoint_dir = f"{run_directory}/checkpoint-{steps}"
        else:
            latest_ckpt = sorted_checkpoint_steps[-1]
            logger.info(f"latest_ckpt: {latest_ckpt}")
            checkpoint_dir = f"{run_directory}/checkpoint-{latest_ckpt}"
    assert checkpoint_dir is not None
    return checkpoint_dir


HF_TOKEN = os.environ["HF_TOKEN"]

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "loguru",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_TOKEN": HF_TOKEN}
    )  # faster model transfers
)


app = modal.App("vllm-server")

API_KEY = "super-secret-key"
MINUTES = 60  # seconds
VLLM_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# MODEL_NAME = "04-26-1-38-728629-ministral-8b-instruct-2410-writing"
MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"


@app.function(
    image=vllm_image,
    gpu="l40s",
    allow_concurrent_inputs=10,
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
        model_id=MODEL_NAME,
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
