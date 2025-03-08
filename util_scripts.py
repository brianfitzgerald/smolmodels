import fire
import os
import shutil
import modal

from loguru import logger

from scripts.modal_definitons import MODEL_WEIGHTS_VOLUME


def clean_runs_folder(parent_folder="runs"):
    """
    Remove run folders from runs that might have crashed.
    """
    folders = [
        os.path.join(parent_folder, folder)
        for folder in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, folder))
    ]

    for folder in folders:
        logger.info(f"Folder: {folder}")
        items = os.listdir(folder)
        files = [item for item in items if os.path.isfile(os.path.join(folder, item))]
        subdirs = [item for item in items if os.path.isdir(os.path.join(folder, item))]
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            logger.info(f"File: {file_name}, Extension: {file_extension}")

        if len(files) == 1 and files[0] == "wrapper_config.json" and len(subdirs) == 0:
            logger.info(f"Deleting folder: {folder}")
            shutil.rmtree(folder)

def download_dataset(dataset_name: str):
    volume = MODEL_WEIGHTS_VOLUME
    path = f"dataset_files/{dataset_name}.parquet"
    logger.info(f"Downloading dataset to {path}")
    local_path = f"dataset_files/{dataset_name}.parquet"
    with open(local_path, "wb") as f:
        volume.read_file_into_fileobj(path, f)
    pass

if __name__ == "__main__":
    fire.Fire()
