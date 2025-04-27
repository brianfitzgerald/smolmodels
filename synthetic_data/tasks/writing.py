import asyncio
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, TypedDict

import polars as pl
import tiktoken
from datasets import Dataset
from huggingface_hub import snapshot_download
from loguru import logger
from pydantic import BaseModel

from synthetic_data.generation import (
    GenWrapperArgs,
    GenerationWrapper,
    get_generation_wrapper,
)
from synthetic_data.gutenberg_parser import super_cleaner
from synthetic_data.prompts import (
    format_classify_fiction_prompt,
    format_writing_backtranslation_prompt,
    tags_to_instruction,
)
from synthetic_data.screenplay_parser import ScreenplayParser
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import Conversation, DatasetFormat
from synthetic_data.tasks import RunMode
from synthetic_data.creative_writing_bench import (
    CreativeWritingBench,
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

    def load_custom(self, dataset_root_path: str) -> Dataset:
        return Dataset.from_parquet(os.path.join(dataset_root_path, "epubs.parquet"))  # type: ignore


async def score_writing(
    completions: list[str],
    prompts: list[str],
    bench: CreativeWritingBench,
    judge_generator: GenerationWrapper,
) -> list[dict[str, float]]:
    """Perform inference with a judge model and return a list of score dictionaries."""
    judge_convs: list[Conversation] = [
        [
            {
                "role": "user",
                "content": bench.format_prompt(prompt, completion),
            }
        ]
        for prompt, completion in zip(prompts, completions)
    ]
    logger.info(
        f"Judging {len(judge_convs)} completions with {judge_generator.args.model_id}"
    )
    judge_completions = await judge_generator.generate(judge_convs)

    if any([x is None for x in judge_completions]):
        logger.warning("Some judge completions were None")
        return []

    score_dicts = [bench.parse_judge_scores(score) for score in judge_completions]
    return score_dicts


async def _generate_and_score(
    generator: GenerationWrapper,
    input_rows: List[Dict],
    bench: CreativeWritingBench,
    judge_generator: GenerationWrapper,
):
    input_convs: list[Conversation] = [
        [{"role": "user", "content": row["instruction"]}] for row in input_rows
    ]
    logger.info(
        f"Generating {len(input_convs)} completions with {generator.args.model_id}"
    )
    completions = await generator.generate(input_convs)

    scores_formatted = await score_writing(
        completions, [row["instruction"] for row in input_rows], bench, judge_generator
    )
    assert scores_formatted is not None, "scores_formatted is None"

    return [
        {
            "scores": s,
            "completion": c,
            "instruction": r["instruction"],
            "model_id": generator.args.model_id,
            "prompt_id": i,
        }
        for i, (r, s, c) in enumerate(zip(input_rows, scores_formatted, completions))
    ]


class BacktranslateBestOfN(BaseTask):
    """
    Take backtranslated snippets, generate completions, and score them. Return a set of N completions with scores.
    """

    output_dataset_name = "backtranslate_best_of_n"
    dataset_columns = ["completion", "instruction", "scores", "model_id"]
    seed_data_format = DatasetFormat.PARQUET
    seed_data_location = "gutenberg_backtranslate_from_txt"
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(self, run_mode: RunMode = "cli") -> None:
        super().__init__(run_mode)
        self.bench = None
        self.generators = [
            get_generation_wrapper("gemini-2.5-flash"),
            get_generation_wrapper("gpt-4o-mini"),
            get_generation_wrapper("gpt-4o"),
        ]
        self.judge_generator = get_generation_wrapper("gemini-2.0-flash")

    async def generate(
        self, generation_wrapper: GenerationWrapper, input_rows: List[Dict]
    ) -> list[dict]:
        if self.bench is None:
            self.bench = CreativeWritingBench(self.run_mode)
        # Launch generation and scoring for each generator concurrently
        results = await asyncio.gather(
            *[
                _generate_and_score(
                    generator, input_rows, self.bench, self.judge_generator
                )
                for generator in self.generators
            ]
        )

        all_results = [item for sublist in results for item in sublist]
        all_results = [x for x in all_results if x is not None]
        return all_results
