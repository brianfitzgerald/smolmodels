from typing import Iterator
import fire
import os
import shutil
import subprocess
import concurrent.futures


from loguru import logger

from scripts.modal_definitons import MODEL_WEIGHTS_VOLUME


def _convert_epub_to_txt(epub_path: str, out_dir: str) -> str:
    base = os.path.basename(epub_path)
    txt_path = os.path.join(out_dir, base + ".txt")
    command = ["pandoc", epub_path, "-t", "plain", "-o", txt_path]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Converted: {epub_path} -> {txt_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {epub_path}: {e}")
    return epub_path


def _find_epub_files(root_dir: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".epub"):
                yield os.path.join(dirpath, filename)


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
    """
    Download a dataset from the modal datasets folder.
    """
    volume = MODEL_WEIGHTS_VOLUME
    path = f"dataset_files/{dataset_name}.parquet"
    logger.info(f"Downloading dataset to {path}")
    local_path = f"dataset_files/{dataset_name}.parquet"
    with open(local_path, "wb") as f:
        volume.read_file_into_fileobj(path, f)
    pass


def convert_epubs_to_txt(root_dir: str, out_dir: str = "~/Documents/txt"):
    """
    Convert all epub files in the root_dir to txt.
    """
    root_dir = os.path.expanduser(root_dir)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    epub_files = list(_find_epub_files(root_dir))
    logger.info(f"Found {len(epub_files)} EPUB files in {root_dir}.")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_convert_epub_to_txt, epub, out_dir): epub
            for epub in epub_files
        }
        for future in concurrent.futures.as_completed(futures):
            epub_file = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(f"{epub_file} generated an exception: {exc}")


if __name__ == "__main__":
    fire.Fire()
