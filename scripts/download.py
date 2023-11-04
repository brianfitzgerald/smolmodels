import os
import sys
from pathlib import Path
from typing import Optional
import fire

import torch
from lightning_utilities.core.imports import RequirementCache

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")


def download_from_hub(
    repo_id: Optional[str] = None, access_token: Optional[str] = os.getenv("HF_TOKEN")
) -> None:
    if repo_id is None:

        print("Please specify a repo ID")
        return

    from huggingface_hub import snapshot_download

    download_files = ["tokenizer*", "generation_config.json"]
    if not _SAFETENSORS_AVAILABLE:
        raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
    download_files.append("*.safetensors")

    directory = Path("checkpoints") / repo_id
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )

    # convert safetensors to PyTorch binaries
    from safetensors import SafetensorError # type: ignore
    from safetensors.torch import load_file as safetensors_load

    print("Converting .safetensor files to PyTorch binaries (.bin)")
    for safetensor_path in directory.glob("*.safetensors"):
        bin_path = safetensor_path.with_suffix(".bin")
        try:
            result = safetensors_load(safetensor_path)
        except SafetensorError as e:
            raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
        print(f"{safetensor_path} --> {bin_path}")
        torch.save(result, bin_path)
        os.remove(safetensor_path)


if __name__ == "__main__":
    fire.Fire(download_from_hub)