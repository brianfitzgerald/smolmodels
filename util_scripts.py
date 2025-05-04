from typing import Iterator
import fire
import os
import shutil
import subprocess
import concurrent.futures
import polars as pl

from openai import OpenAI

from loguru import logger

from scripts.modal_definitions import MODEL_WEIGHTS_VOLUME


def _convert_epub_to_txt(epub_path: str, out_dir: str) -> str:
    base = os.path.basename(epub_path)
    txt_path = os.path.join(out_dir, base + ".txt")
    command = ["pandoc", epub_path, "-t", "plain", "-o", txt_path]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {epub_path}: {e}")

    txt_file_contents = open(txt_path).read()
    return txt_file_contents


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


def upload_dataset(dataset_name: str):
    """
    Upload a dataset to the modal datasets folder.
    """
    volume = MODEL_WEIGHTS_VOLUME
    path = f"dataset_files/{dataset_name}.parquet"
    logger.info(f"Uploading dataset to {path}")
    with volume.batch_upload(force=True) as upload:
        logger.info(f"Uploading {path}")
        upload.put_file(path, path)
    logger.info(f"Uploaded dataset to {path}")


def convert_epubs_to_txt(root_dir: str, out_dir: str):
    """
    Convert all epub files in the root_dir to txt.
    Once converted, use the gutenberg_dev.ipynb notebook to split the txt files into chunks.
    """
    root_dir = os.path.expanduser(root_dir)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    epub_files = list(_find_epub_files(root_dir))
    logger.info(f"Found {len(epub_files)} EPUB files in {root_dir}.")
    out_rows = []
    n_generated, n_total = 0, len(epub_files)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_convert_epub_to_txt, epub, out_dir): epub
            for epub in epub_files
        }
        for future in concurrent.futures.as_completed(futures):
            epub_file = futures[future]
            try:
                txt_file_contents = future.result()
                out_rows.append(
                    {
                        "title": os.path.basename(epub_file),
                        "text": txt_file_contents,
                    }
                )
            except Exception as exc:
                logger.error(f"{epub_file} generated an exception: {exc}")
            n_generated += 1
            logger.info(f"Converted {n_generated}/{n_total} EPUB files.")

    out_rows_pl = pl.from_dicts(out_rows)
    out_rows_pl.write_parquet(os.path.join("dataset_files", "epubs.parquet"))


def _get_model_id(client: OpenAI):
    all_models = client.models.list()

    logger.info(f"Available models: {[x.id for x in all_models.data]}")
    print(all_models)
    first_model_id = all_models.data[0].id
    model_id = first_model_id
    logger.info(f"Selected model: {model_id}")
    return model_id


def test_openai_api(
    msg: str = "Write a story told entirely through a series of brief correspondences: telegrams or letters or emails between two characters. The correspondence should span several months or years, and reveal a gradually unfolding plot. Use distinct voices for each character, and include details that provide insight into their personalities and motivations. The story should build to an emotional climax, and the final letter should provide a satisfying resolution. The setting is a lighthouse keeper writing to his mother. He is working class and Scottish. He is struggling with the isolation of his posting. Write naturally and without cliches.",
    base_url: str = "https://brianfitzgerald--vllm-server-serve.modal.run",
):
    """
    Test the OpenAI API.
    """
    oai_client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="super-secret-key",
    )
    model_id = _get_model_id(oai_client)
    response = oai_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": msg},
        ],
        max_tokens=2048,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    fire.Fire()
