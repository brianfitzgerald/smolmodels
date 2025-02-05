import os
from pathlib import Path
import fire
import uvicorn
from modal import Volume
from modal.volume import FileEntryType
from loguru import logger
import subprocess

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    ServerSettings,
    ModelSettings,
)


def convert_hf_to_gguf(path: str):
    command = [
        "python",
        "scripts/convert_hf_to_gguf.py",
        f"{path}/",
        "--outfile",
        f"{path}/model.gguf",
    ]

    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    logger.info(result.stdout)

    return result.returncode


def main(
    run: str = "02-02-20-10-185429-llama-3.2-3b-instruct-playwright-gutenberg-conv",
    checkpoint: str = "checkpoint-1034",
):
    vol = Volume.from_name("model-weights")
    local_dir = f"runs/{run}/{checkpoint}"

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Copy files from Modal
    for file in vol.iterdir(f"runs/{run}/{checkpoint}"):
        if file.type == FileEntryType.FILE:
            rel_path = file.path
            if os.path.exists(rel_path):
                logger.info(f"Skipping {file.path} as it already exists locally.")
                continue
            logger.info(f"Downloading {file.path} to {rel_path}")
            Path(rel_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(rel_path, "wb") as file_obj:
                    vol.read_file_into_fileobj(file.path, file_obj)
            finally:
                logger.info(f"Removing {rel_path} as download was interrupted")
                os.remove(rel_path)

    logger.info("Converting HuggingFace model to GGUF format")
    convert_hf_to_gguf(local_dir)

    server_settings = ServerSettings()
    model_settings = [ModelSettings(model=f"{local_dir}/model.gguf")]
    assert server_settings is not None
    assert model_settings is not None
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )

    uvicorn.run(
        app,
        host=os.getenv("HOST", server_settings.host),
        port=int(os.getenv("PORT", server_settings.port)),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
    )


if __name__ == "__main__":
    fire.Fire(main)
