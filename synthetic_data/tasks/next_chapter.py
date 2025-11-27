import re

import tiktoken
from datasets import Dataset
from loguru import logger
from synthetic_data.generation import GenerationWrapper
from synthetic_data.gutenberg_parser import super_cleaner
from synthetic_data.tasks import BaseTaskV1, RunMode, get_gutenberg_subset
from synthetic_data.utils import Conversation, DatasetFormat

# Pattern to detect chapter headings
CHAPTER_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"CHAPTER\s+[IVXLCDM\d]+\.?|"  # CHAPTER I, CHAPTER 1, etc.
    r"Chapter\s+[IVXLCDMivxlcdm\d]+\.?|"  # Chapter I, Chapter 1, etc.
    r"PART\s+[IVXLCDM\d]+\.?|"  # PART I, PART 1, etc.
    r"Part\s+[IVXLCDMivxlcdm\d]+\.?|"  # Part I, Part 1, etc.
    r"BOOK\s+[IVXLCDM\d]+\.?|"  # BOOK I, BOOK 1, etc.
    r"Book\s+[IVXLCDMivxlcdm\d]+\.?|"  # Book I, Book 1, etc.
    r"ACT\s+[IVXLCDM\d]+\.?|"  # ACT I, ACT 1, etc.
    r"Act\s+[IVXLCDMivxlcdm\d]+\.?|"  # Act I, Act 1, etc.
    r"SCENE\s+[IVXLCDM\d]+\.?|"  # SCENE I, SCENE 1, etc.
    r"Scene\s+[IVXLCDMivxlcdm\d]+\.?|"  # Scene I, Scene 1, etc.
    r"[IVXLCDM]+\.|"  # Roman numeral alone: I., II., etc.
    r"\d+\."  # Numeric: 1., 2., etc.
    r")\s*$",
    re.MULTILINE,
)

MAX_TOKENS = 2048
MIN_NEXT_CHUNK_TOKENS = 256


def is_chapter_heading(paragraph: str) -> bool:
    """Check if a paragraph looks like a chapter heading."""
    stripped = paragraph.strip()
    # Short paragraphs that match chapter pattern
    if len(stripped) < 100 and CHAPTER_HEADING_PATTERN.match(stripped):
        return True
    # Also check for all-caps short lines that look like headings
    if len(stripped) < 80 and stripped.isupper() and len(stripped.split()) <= 6:
        return True
    return False


def find_long_prefix_chunks(
    paragraphs: list[str],
    encoder: tiktoken.Encoding,
    max_tokens: int = MAX_TOKENS,
    min_next_chunk_tokens: int = MIN_NEXT_CHUNK_TOKENS,
) -> list[tuple[str, str]]:
    """
    Find (prefix, next_chunk) pairs where prefix accumulates paragraphs
    up to max_tokens or until a chapter heading is found.
    Both prefix and next_chunk are capped at max_tokens.

    Returns a list of (prefix, next_chunk) tuples.
    """
    if len(paragraphs) < 2:
        return []

    results: list[tuple[str, str]] = []
    i = 0

    while i < len(paragraphs):
        # Accumulate paragraphs for prefix
        prefix_paragraphs: list[str] = []
        prefix_tokens = 0

        while i < len(paragraphs):
            para = paragraphs[i]

            # Check if this paragraph is a chapter heading
            if prefix_paragraphs and is_chapter_heading(para):
                # Stop accumulating - this heading starts a new section
                break

            # Check if adding this paragraph would exceed max tokens
            para_tokens = len(encoder.encode(para + "\n\n"))
            if prefix_tokens > 0 and prefix_tokens + para_tokens > max_tokens:
                # Stop accumulating - we've hit the token limit
                break

            prefix_paragraphs.append(para)
            prefix_tokens += para_tokens
            i += 1

        # Now accumulate next_chunk paragraphs
        next_chunk_paragraphs: list[str] = []
        next_chunk_tokens = 0
        j = i

        while j < len(paragraphs):
            para = paragraphs[j]

            # Check if adding this paragraph would exceed max tokens
            para_tokens = len(encoder.encode(para + "\n\n"))
            if next_chunk_tokens > 0 and next_chunk_tokens + para_tokens > max_tokens:
                # Stop accumulating - we've hit the token limit
                break

            # Stop at chapter heading (after getting minimum content)
            if next_chunk_paragraphs and is_chapter_heading(para):
                if next_chunk_tokens >= min_next_chunk_tokens:
                    break

            next_chunk_paragraphs.append(para)
            next_chunk_tokens += para_tokens
            j += 1

            # Stop if we have enough content for next_chunk
            if next_chunk_tokens >= min_next_chunk_tokens:
                # Check if next paragraph is a chapter heading
                if j < len(paragraphs) and is_chapter_heading(paragraphs[j]):
                    break

        # Create the pair if we have both parts
        if prefix_paragraphs and next_chunk_paragraphs:
            prefix = "\n\n".join(prefix_paragraphs)
            next_chunk = "\n\n".join(next_chunk_paragraphs)
            results.append((prefix, next_chunk))

        # Move to next section (skip the next_chunk we just processed)
        i = j

    return results


class GutenbergSummaryContinuation(BaseTaskV1):
    output_dataset_name = "gutenberg_summary_continuation"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.CUSTOM
    seed_data_location = "gutenberg"  # Dummy value for CUSTOM format
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(self, run_mode: RunMode) -> None:
        super().__init__(run_mode)
        self.encoder = tiktoken.get_encoding("o200k_base")

    def load_custom(self, dataset_root_path: str) -> Dataset:
        shards_pl = get_gutenberg_subset(2)
        logger.info(f"Loaded {len(shards_pl)} rows from Gutenberg dataset")
        return Dataset.from_polars(shards_pl)

    async def preprocess_row(self, row: dict) -> list[dict]:
        text = row["text"]
        paragraphs = super_cleaner(text)

        metadata = {
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "id": row.get("id", ""),
        }

        # Find long prefix chunks (up to 8k tokens or chapter boundary)
        chunk_pairs = find_long_prefix_chunks(paragraphs, self.encoder)

        if not chunk_pairs:
            logger.warning(f"No valid chunks found for {metadata['title']}")
            return []

        rows_out = []
        for prefix, next_chunk in chunk_pairs:
            rows_out.append(
                {
                    "prefix": prefix,
                    "next_chunk": next_chunk,
                    **metadata,
                }
            )
        return rows_out

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        """Format the prefix text for summarization."""
        conv_out = []
        for sample in batch:
            conv: Conversation = [
                {
                    "role": "system",
                    "content": "You are an expert at creating concise, informative summaries. Summarize the following narrative passage, capturing key events, character actions, themes, and important details. Keep the summary focused and under 200 words.",
                },
                {"role": "user", "content": sample["prefix"]},
            ]
            conv_out.append(conv)
        return conv_out

    def format_output_rows(
        self, completions: list[str], input_rows: list[dict]
    ) -> list:
        out_rows = []
        for completion, row in zip(completions, input_rows):
            out_rows.append(
                {
                    "prefix": row["prefix"],
                    "summary": completion,
                    "next_chunk": row["next_chunk"],
                    "title": row.get("title", ""),
                    "author": row.get("author", ""),
                    "id": row.get("id", ""),
                }
            )
        return out_rows

    async def generate(
        self, generation_wrapper: GenerationWrapper, input_rows: list[dict]
    ) -> list[dict]:
        try:
            summary_convs = self.format_input_conversation(input_rows)
            completions = await generation_wrapper.generate(summary_convs)
            return self.format_output_rows(completions, input_rows)
        except TimeoutError:
            logger.error("Timeout error processing batch")
            return []
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []
