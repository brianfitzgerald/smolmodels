import functools
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple, TypedDict

import polars as pl
import tiktoken
from datasets import Dataset
from huggingface_hub import snapshot_download
from loguru import logger
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from tqdm import tqdm

from synthetic_data.generation import GenWrapperArgs
from synthetic_data.gutenberg_parser import DIALOGUE_REGEX, super_cleaner
from synthetic_data.judgemark import TASK_PROMPT
from synthetic_data.prompts import (
    format_gutenberg_backtranslation_prompt,
    format_gutenberg_followup_prompt,
)
from synthetic_data.screenplay_parser import ScreenplayParser
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import Conversation, DatasetFormat, dictl
from synthetic_data.writing_judge import (
    JUDGING_CRITERIA,
    format_gutenberg_judge_prompt,
    parse_scores,
)


@dataclass
class SceneRow:
    name: str
    text_summary: str


class ScreenplaySummarize(BaseTask):
    output_dataset_name = "screenplay_scenes_summarized_full"
    dataset_columns = ["completions", "test_results", "name"]
    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(self) -> None:
        self.in_rows_batch = []
        self.max_samples = 50000

    def load_custom(self):
        return Dataset.from_dict({})
        # scripts_corpus_path = kagglehub.dataset_download(
        #     "veeralakrishna/imsdb-movie-scripts"
        # )
        # scripts_pqt_path = os.path.join(scripts_corpus_path, "movie_scripts.parquet")
        # dataset: Dataset = Dataset.from_parquet(scripts_pqt_path)  # type: ignore
        # return dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        def process_row(row):
            new_rows = []
            movie_title = row["Movie"]
            parser = ScreenplayParser(row["Script"])
            parser.parse()
            if len(parser.scenes) < 20 or len(parser.character_line_counts) < 20:
                return new_rows
            for scene in parser.scenes:
                new_rows.append(
                    {
                        "name": movie_title,
                        "scene": ScreenplayParser.format_conversation(scene),
                    }
                )
            return new_rows

        all_new_rows = []
        with ThreadPoolExecutor() as executor:
            for result in tqdm(
                executor.map(process_row, dataset),
                desc="Splitting scenes",
                total=len(dataset),
            ):
                all_new_rows.extend(result)

        dataset = Dataset.from_list(all_new_rows)
        dataset = dataset.filter(lambda x: x["scene"] != "")
        dataset = dataset.shuffle(seed=42)
        # dataset = dataset.select(range(self.max_samples))
        return dataset

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        conv_out = []
        for sample in samples_in:
            scene = sample["scene"]
            conv: Conversation = [
                {
                    "role": "system",
                    "content": "Summarize the content of the following screenplay scene. Describe the actions of the characters and the contents of the scene. Start responding with the first character's actions or dialogue.",
                },
                {"role": "user", "content": scene},
            ]
            conv_out.append(conv)
            self.in_rows_batch.append(sample)
        return conv_out

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, row in zip(completions, self.in_rows_batch):
            row["summary"] = completion
            out_rows.append(row)
        self.in_rows_batch = []
        return out_rows


PROMPT_PREFIX_PATTERN = re.compile(r"^(?:Compose|Write) a chapter$")
TITLE_PATTERN = re.compile(r"\[([^]]+)\]\s+(.+?)\s+--\s+(\S+)")

MAX_LEN_TOKENS = 1536


class ProcessedRow(TypedDict):
    prompt: str
    text: str
    category: str
    author: str
    title: str
    encoded_length: int


def remove_last_paragraph(text: str):
    paragraphs = text.split("\n\n")
    paragraphs = paragraphs[:-1]
    return "\n\n".join(paragraphs)


def _process_gutenberg_extraction_row(
    row: dict, encoder: tiktoken.Encoding
) -> ProcessedRow:
    prompt = row["prompt"]
    prompt = PROMPT_PREFIX_PATTERN.sub("", prompt).strip()
    match = TITLE_PATTERN.match(row["source"])
    category, author, title = "", "", ""
    if match is None:
        raise ValueError(f"Could not parse title from prompt: {prompt}")
    category, author, title = match.groups()
    text = row["chosen"][-1]["content"]
    text_encoded = encoder.encode(text)
    encoded_len = len(text_encoded)
    if len(text_encoded) > MAX_LEN_TOKENS:
        text_encoded = text_encoded[:MAX_LEN_TOKENS]
        text = encoder.decode(text_encoded)
        text = remove_last_paragraph(text)
    return {
        "prompt": prompt,
        "text": text,
        "category": category,
        "author": author,
        "title": title,
        "encoded_length": encoded_len,
    }


def _format_gutenberg_conv(sample: dict) -> Conversation:
    return [
        {
            "role": "system",
            "content": "Extract the dialogue, actions, and descriptions from the conversation given by the user.",
        },
        {"role": "user", "content": sample["text"]},
    ]


class SceneElementType(Enum):
    SCENE_HEADING = "scene_heading"
    ACTION = "action"
    DIALOGUE = "dialogue"
    TRANSITION = "transition"


class SceneElement(BaseModel):
    type: SceneElementType
    content: str
    character: str | None = None


class Output(BaseModel):
    items: List[SceneElement]


class GutenbergExtraction(BaseTask):
    """
    Extract dialogue and actions from Gutenberg snippets.
    """

    output_dataset_name = "screenplay_scenes_summarized_full"
    dataset_columns = ["completions", "test_results", "name"]
    seed_data_format = DatasetFormat.HF_DATASET
    output_dataset_format = DatasetFormat.PARQUET
    seed_data_location = (
        "sam-paech/gutenberg3-generalfiction-scifi-fantasy-romance-adventure-dpo"
    )

    gen_wrapper_args_override = GenWrapperArgs(response_format=Output)

    def __init__(self) -> None:
        self.in_rows_batch = []
        self.tiktoken_encoder = tiktoken.get_encoding("o200k_base")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        map_fn = functools.partial(
            _process_gutenberg_extraction_row, encoder=self.tiktoken_encoder
        )
        dataset = dataset.map(map_fn)
        return dataset

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        self.in_rows_batch = samples_in
        # TODO length splitting
        return [_format_gutenberg_conv(sample) for sample in samples_in]

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, row in zip(completions, self.in_rows_batch):
            json_str = Output.model_validate(json.loads(completion)).model_dump_json()
            out_rows.append({**row, "output": json_str})
        self.in_rows_batch = []
        return out_rows


def _format_annotate_conv(text: str) -> Sequence[ChatCompletionMessageParam]:
    prompt_formatted = TASK_PROMPT.replace("[TEST MODEL RESPONSE]", text).replace(
        "[TEST MODEL RESPONSE END]", ""
    )

    return [
        {"role": "user", "content": prompt_formatted},
    ]


class WritingRewardAnnotate(BaseTask):
    output_dataset_name = "writing_reward_annotated"
    dataset_columns = ["completions", "test_results", "name"]
    seed_data_format = DatasetFormat.HF_DATASET
    output_dataset_format = DatasetFormat.HF_DATASET
    seed_data_location = (
        "sam-paech/gutenberg3-generalfiction-scifi-fantasy-romance-adventure-dpo"
    )

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        in_batch = []
        samples_out = []
        for sample in samples_in:
            chosen_text = sample["chosen"][-1]["content"]
            rejected_text = sample["rejected"][-1]["content"]
            samples_out.append(_format_annotate_conv(chosen_text))
            samples_out.append(_format_annotate_conv(rejected_text))
            in_batch.extend([sample, sample])
        self.in_rows_batch = in_batch
        return samples_out

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, row in zip(completions, self.in_rows_batch):
            out_rows.append({**row, "output": completion})
        self.in_rows_batch = []
        return out_rows


def _contains_dialogue(text):
    return bool(DIALOGUE_REGEX.search(text))


def _count_sentences(text):
    return len(re.findall(r"[.!?]", text))


def find_valid_paragraph_chunks(
    lst: list[str],
    encoder: tiktoken.Encoding,
    min_chunk_size_tokens: int = 600,
) -> list[list[str]]:
    """
    Find chunks of consecutive paragraphs with dialogue and at least 2 sentences,
    and at least min_chunk_size_tokens tokens.
    """
    chunks, current_chunk = [], []
    current_chunk_tokens = 0
    for item in lst:
        if (
            item != "[deleted]"
            and _contains_dialogue(item)
            and _count_sentences(item) >= 2
        ):
            item_tokens = len(encoder.encode(item))
            current_chunk.append(item)
            current_chunk_tokens += item_tokens
        else:
            if current_chunk_tokens >= min_chunk_size_tokens:
                chunks.append(current_chunk)
            current_chunk = []
            current_chunk_tokens = 0

    # handle the last chunk
    if current_chunk_tokens >= min_chunk_size_tokens:
        chunks.append(current_chunk)

    return chunks


def _get_paragraph_chunks(row: dict, encoder: tiktoken.Encoding):
    text = row["text"]

    # clean and extract paragraphs
    paragraphs = super_cleaner(text)

    # find chunks of consecutive paragraphs with dialogue and at least 3 sentences
    valid_chunks = find_valid_paragraph_chunks(paragraphs, encoder)

    # filter out chunks that are too long or too short
    out = []
    for chunk in valid_chunks:
        tokens = encoder.encode(" ".join(chunk))
        logger.info(f"Chunk length: {len(tokens)}")
        if len(tokens) < 1000:
            out.append(chunk)
    return out


def _get_gutenberg_subset(n_shards: int = 1) -> pl.DataFrame:
    gutenberg_location = snapshot_download("SaylorTwift/Gutenberg", repo_type="dataset")
    files = os.listdir(os.path.join(gutenberg_location, "data"))
    files.sort()
    out_pl = None
    for shard_idx in range(n_shards):
        file = files[shard_idx]
        df = pl.read_parquet(os.path.join(gutenberg_location, "data", file))
        if out_pl is None:
            out_pl = df
        else:
            out_pl = out_pl.vstack(df)
    assert out_pl is not None, "could not find any shards"
    return out_pl


class GutenbergBacktranslation(BaseTask):
    """
    Generate a high quality prompt from a Gutenberg chunk.
    """

    output_dataset_name = "gutenberg_backtranslate"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(self) -> None:
        self.tiktoken_encoder = tiktoken.get_encoding("o200k_base")

    def load_custom(self) -> Dataset:
        shards_pl = _get_gutenberg_subset(2)
        logger.info(f"Loaded {len(shards_pl)} rows from Gutenberg dataset")
        return Dataset.from_polars(shards_pl)

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.shuffle(seed=64)
        return dataset

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        all_chunks: List[Tuple[List[str], Dict]] = [
            (
                _get_paragraph_chunks(sample, self.tiktoken_encoder),
                {
                    "title": sample["title"],
                    "author": sample["author"],
                    "id": sample["id"],
                },
            )
            for sample in samples_in
        ]
        paragraphs: List[Tuple[str, Dict]] = []
        for chunks, metadata in all_chunks:
            if len(chunks) == 0:
                logger.warning(f"No chunks found for {metadata['title']}")
                continue
            for chunk in chunks:
                paragraphs.append(("\n\n".join(chunk), metadata))
        self.metadata = [chunk_metadata for _, chunk_metadata in paragraphs]
        self.paragraphs = [chunk for chunk, _ in paragraphs]
        return [format_gutenberg_backtranslation_prompt(p) for p, _ in paragraphs]

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, metadata, paragraph in zip(
            completions, self.metadata, self.paragraphs
        ):
            out_rows.append(
                {
                    "instruction": completion,
                    "paragraph": paragraph,
                    **metadata,
                }
            )
        return out_rows


class GutenbergBacktranslationFromTxt(GutenbergBacktranslation):
    output_dataset_name = "gutenberg_backtranslate_from_txt"

    def load_custom(self) -> Dataset:
        txt_dir = os.path.expanduser("~/Documents/txt")
        txt_files = list(glob.glob(os.path.join(txt_dir, "*.txt")))
        txt_file_contents = [
            {
                "title": os.path.basename(txt_file),
                "text": open(txt_file).read(),
                "id": "",
                "author": "",
                "category": "",
            }
            for txt_file in txt_files
        ]
        return Dataset.from_list(txt_file_contents)


class WritingScoreAnnotate(BaseTask):
    output_dataset_name = "gutenberg_score_annotated"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.PARQUET
    seed_data_location = "gutenberg_backtranslate"
    output_dataset_format = DatasetFormat.PARQUET

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        self.samples_in = samples_in
        self.sample_idxs = []
        formatted_convs = []
        for i, sample in enumerate(samples_in):
            for j in range(len(JUDGING_CRITERIA)):
                formatted_convs.append(
                    format_gutenberg_judge_prompt(
                        sample["instruction"], sample["paragraph"], JUDGING_CRITERIA[j]
                    )
                )
            self.sample_idxs.extend([i] * len(formatted_convs))

        return formatted_convs

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, sample_idx in zip(completions, self.sample_idxs):
            sample = self.samples_in[sample_idx]
            scores = parse_scores(completion)
            out_rows.append({**sample, "scores": scores})
        return out_rows


class GutenbergFollowUp(BaseTask):
    output_dataset_name = "gutenberg_followup"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.CUSTOM
    output_dataset_format = DatasetFormat.PARQUET

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        samples_in = dictl(batch)
        self.samples_in = samples_in
        return [
            format_gutenberg_followup_prompt(sample["paragraph"], sample["instruction"])
            for sample in samples_in
        ]

    def format_output_rows(self, completions: List[str]) -> List:
        out_rows = []
        for completion, sample in zip(completions, self.samples_in):
            out_rows.append({**sample, "output": completion})
        return out_rows
