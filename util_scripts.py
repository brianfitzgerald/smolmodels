import fire
import os
import shutil

from loguru import logger


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


if __name__ == "__main__":
    fire.Fire()
