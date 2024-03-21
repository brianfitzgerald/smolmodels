import logging
import os
from typing import Dict, Union, Tuple

import fasttext
import requests

logger = logging.getLogger(__name__)
models = {"low_mem": None, "high_mem": None}
FTLANG_CACHE = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")


def download_model(name: str) -> str:
    target_path = os.path.join(FTLANG_CACHE, name)
    if not os.path.exists(target_path):
        logger.info(f"Downloading {name} model ...")
        url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{name}"  # noqa
        os.makedirs(FTLANG_CACHE, exist_ok=True)
        with open(target_path, "wb") as fp:
            response = requests.get(url)
            fp.write(response.content)
        logger.info(f"Downloaded.")
    return target_path


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            model_path = download_model("lid.176.ftz")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model # type: ignore
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            model_path = download_model("lid.176.bin")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model # type: ignore
        return model


def detect(text: str, low_memory=False) -> Dict[str, Union[str, float]]:
    model = get_or_load_model(low_memory)
    labels, scores = model.predict(text)
    label = labels[0].replace("__label__", '') # type: ignore
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }

SUPPORTED_LANGUAGES = ["en"]

def detect_if_supported_language(
    text: str
) -> Tuple[bool, str]:
    cleaned_text = "".join(text.splitlines())

    low_mem_detect = detect(cleaned_text, low_memory=True)
    lang_id_low_mem = low_mem_detect["lang"]

    if lang_id_low_mem in SUPPORTED_LANGUAGES:
        return True, lang_id_low_mem

    high_mem_detect = detect(cleaned_text, low_memory=False)
    lang_id_high_mem: str = high_mem_detect["lang"] # type: ignore

    if lang_id_high_mem != lang_id_low_mem:
        logger.info(
            f"Different language detected; low_mem: {low_mem_detect}; high_mem: {high_mem_detect} for prompt [{text}]"
        )

    if lang_id_high_mem in SUPPORTED_LANGUAGES:
        return True, lang_id_high_mem

    logger.info(f"Unsupported language detected: {high_mem_detect} for prompt [{text}]")
    return False, lang_id_high_mem
