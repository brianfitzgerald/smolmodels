import json
import os
import re
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

from synthetic_data.generation import GenWrapperArgs, GenerationWrapper
from synthetic_data.gutenberg_parser import DIALOGUE_REGEX, super_cleaner
from synthetic_data.judgemark import TASK_PROMPT
from synthetic_data.prompts import (
    format_classify_fiction_prompt,
    format_writing_backtranslation_prompt,
    format_gutenberg_followup_prompt,
    tags_to_instruction,
)
from synthetic_data.screenplay_parser import ScreenplayParser
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import Conversation, DatasetFormat
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

    async def preprocess_row(self, row: dict) -> list[dict]:
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

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        samples_in = batch
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

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
        out_rows = []
        for completion, row in zip(completions, input_rows):
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

    async def preprocess_row(self, row: dict) -> list[dict]:
        return [_process_gutenberg_extraction_row(row, self.tiktoken_encoder)]  # type: ignore

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        samples_in = batch
        self.in_rows_batch = samples_in
        return [_format_gutenberg_conv(sample) for sample in samples_in]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
        out_rows = []
        for completion, row in zip(completions, input_rows):
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

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        samples_in = batch
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

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
        out_rows = []
        for completion, row in zip(completions, input_rows):
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
    min_chunk_size_tokens: int = 1500,
) -> list[list[str]]:
    chunks, current_chunk = [], []
    current_chunk_tokens = 0

    for item in lst:
        if item != "[deleted]":
            item_tokens = len(encoder.encode(item))

            if current_chunk and current_chunk_tokens >= min_chunk_size_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = 0
            else:
                current_chunk.append(item)
                current_chunk_tokens += item_tokens

    if current_chunk and current_chunk_tokens >= min_chunk_size_tokens:
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
        if len(tokens) < 10000:
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


def extract_fiction_label(text):
    match = re.search(r"\b(Narrative Fiction|Not Narrative Fiction)\b", text)
    if match:
        label_str = match.group(1)
        label = 1 if label_str == "Narrative Fiction" else 0
        return label
    else:
        return None


def extract_tags_from_instruction(text):
    instruction_pattern = r"<instruction>(.*?)</instruction>"
    instruction_match = re.search(instruction_pattern, text, re.DOTALL)

    if not instruction_match:
        return {}

    instruction_content = instruction_match.group(1)
    tag_pattern = r"<(\w+)>\s*(.*?)\s*</\1>"
    matches = re.findall(tag_pattern, instruction_content, re.DOTALL)

    extracted = {}
    for tag, content in matches:
        extracted[tag.strip()] = content.strip()

    return extracted


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

    def load_custom(self, dataset_root_path: str) -> Dataset:
        shards_pl = _get_gutenberg_subset(2)
        logger.info(f"Loaded {len(shards_pl)} rows from Gutenberg dataset")
        return Dataset.from_polars(shards_pl)

    async def preprocess_row(self, row: Dict) -> list[dict]:
        all_chunks: List[Tuple[List[str], Dict]] = [
            (
                _get_paragraph_chunks(row, self.tiktoken_encoder),
                {
                    "title": row.get("title", ""),
                    "author": row.get("author", ""),
                    "id": row.get("id", ""),
                },
            )
        ]
        paragraphs: List[Tuple[str, Dict]] = []
        for chunks, metadata in all_chunks:
            if len(chunks) == 0:
                logger.warning(f"No chunks found for {metadata['title']}")
                continue
            for chunk in chunks:
                paragraphs.append(("\n\n".join(chunk), metadata))
        rows_out = []
        for chunk, metadata in paragraphs:
            rows_out.append(
                {
                    "text": chunk,
                    **metadata,
                }
            )
        return rows_out

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
        out_rows = []
        for completion, row in zip(completions, input_rows):
            tags = extract_tags_from_instruction(completion)
            out_rows.append(
                {
                    "completion": completion,
                    "paragraph": row["text"],
                    **row,
                    "tags": tags,
                    "instruction": tags_to_instruction(tags),
                }
            )
        return out_rows

    async def generate(
        self, generation_wrapper: GenerationWrapper, input_rows: list[dict]
    ) -> list[dict]:
        try:
            filter_convs = [
                format_classify_fiction_prompt(p["text"]) for p in input_rows
            ]
            filter_completions = await generation_wrapper.generate(filter_convs)
            filter_labels = [extract_fiction_label(c) for c in filter_completions]
            logger.info(filter_labels)

            valid_rows = [
                row for row, label in zip(input_rows, filter_labels) if label == 1
            ]
            logger.info(f"Valid rows: {len(valid_rows)}")

            bt_convs = [
                format_writing_backtranslation_prompt(p["text"]) for p in valid_rows
            ]
            completions = await generation_wrapper.generate(bt_convs)
            return self.format_output_rows(completions, valid_rows)
        except TimeoutError:
            logger.error("Timeout error processing batch")
            return []
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []


class GutenbergBacktranslationFromTxt(GutenbergBacktranslation):
    output_dataset_name = "gutenberg_backtranslate_from_txt"
    seed_data_format = DatasetFormat.CUSTOM
    seed_data_location = "epubs"

    def load_custom(self, dataset_root_path: str) -> Dataset:
        return Dataset.from_parquet(os.path.join(dataset_root_path, "epubs.parquet"))  # type: ignore


class WritingScoreAnnotate(BaseTask):
    output_dataset_name = "gutenberg_score_annotated"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.PARQUET
    seed_data_location = "gutenberg_backtranslate"
    output_dataset_format = DatasetFormat.PARQUET

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        samples_in = batch
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

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
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

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        samples_in = batch
        self.samples_in = samples_in
        return [
            format_gutenberg_followup_prompt(sample["paragraph"], sample["instruction"])
            for sample in samples_in
        ]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List:
        out_rows = []
        for completion, sample in zip(completions, input_rows):
            out_rows.append({**sample, "output": completion})
        return out_rows
