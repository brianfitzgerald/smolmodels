import re
from dataclasses import dataclass
import polars as pl
import os
import tiktoken
from datasets import Dataset
from loguru import logger
from synthetic_data.generation import GenerationWrapper
from synthetic_data.gutenberg_parser import super_cleaner
from synthetic_data.tasks import BaseTaskV1, RunMode
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

# Pattern to detect metadata lines (author credits, contents, etc.)
METADATA_PATTERN = re.compile(
    r"^(?:"
    r"By\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|"  # By George Meredith
    r"Contents:?|"  # Contents:
    r"Table\s+of\s+Contents|"
    r"Introduction|"
    r"Preface|"
    r"Foreword|"
    r"Dedication|"
    r"Acknowledgements?|"
    r"Copyright|"
    r"Published\s+by|"
    r"First\s+published|"
    r"All\s+rights\s+reserved|"
    r"ISBN|"
    r"Illustrated\s+by|"
    r"Edited\s+by|"
    r"Translated\s+by"
    r")\s*$",
    re.IGNORECASE,
)

# Minimum requirements for valid narrative content


@dataclass
class NCPConfig:
    """Configuration for chunk extraction."""

    summarize: bool = False
    max_summary_tokens: int = 1024
    max_prefix_tokens: int = 512
    min_next_chunk_tokens: int = 512
    min_paragraph_words: int = 15
    min_prefix_words: int = 30
    min_summary_words: int = 100


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


def is_metadata(paragraph: str) -> bool:
    """Check if a paragraph looks like metadata (author, contents, etc.)."""
    stripped = paragraph.strip()
    # Very short content is likely metadata
    if len(stripped) < 50 and METADATA_PATTERN.match(stripped):
        return True
    return False


def is_valid_narrative_paragraph(paragraph: str, min_words: int = 15) -> bool:
    """
    Check if a paragraph contains valid narrative content.

    A valid narrative paragraph should:
    - Have enough words
    - Not be metadata
    - Not be a chapter heading
    - Contain actual prose (not just dialogue markers or short fragments)
    """
    stripped = paragraph.strip()

    # Check minimum word count
    words = stripped.split()
    if len(words) < min_words:
        return False

    # Check for metadata patterns
    if is_metadata(stripped):
        return False

    # Check for chapter headings
    if is_chapter_heading(stripped):
        return False

    return True


def filter_valid_paragraphs(paragraphs: list[str], min_words: int) -> list[str]:
    """Filter paragraphs to only include valid narrative content."""
    return [p for p in paragraphs if is_valid_narrative_paragraph(p, min_words)]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def find_chunks_with_summary(
    config: NCPConfig,
    paragraphs: list[str],
    encoder: tiktoken.Encoding,
) -> list[tuple[str, str, str]]:
    """
    Find (prefix, summary_content, next_chunk) tuples where:
    - summary_content: longer passage (up to max_summary_tokens) for summarization
    - prefix: shorter passage (up to max_prefix_tokens) extracted from the start
    - next_chunk: the continuation passage

    All outputs are validated to contain reasonable narrative content.

    Returns a list of (prefix, summary_content, next_chunk) tuples.
    """

    # First filter out invalid paragraphs
    valid_paragraphs = filter_valid_paragraphs(paragraphs, config.min_paragraph_words)

    if len(valid_paragraphs) < 2:
        return []

    results: list[tuple[str, str, str]] = []
    i = 0

    while i < len(valid_paragraphs):
        # Accumulate paragraphs for summary_content (longer passage)
        summary_paragraphs: list[str] = []
        summary_tokens = 0

        while i < len(valid_paragraphs):
            para = valid_paragraphs[i]

            # Check if this paragraph is a chapter heading
            if summary_paragraphs and is_chapter_heading(para):
                # Stop accumulating - this heading starts a new section
                break

            # Check if adding this paragraph would exceed max tokens
            para_tokens = len(encoder.encode(para + "\n\n"))
            if (
                summary_tokens > 0
                and summary_tokens + para_tokens > config.max_summary_tokens
            ):
                # Stop accumulating - we've hit the token limit
                break

            summary_paragraphs.append(para)
            summary_tokens += para_tokens
            i += 1

        # Extract shorter prefix from the beginning of summary_content
        prefix_paragraphs: list[str] = []
        prefix_tokens = 0
        for para in summary_paragraphs:
            para_tokens = len(encoder.encode(para + "\n\n"))
            if (
                prefix_tokens > 0
                and prefix_tokens + para_tokens > config.max_prefix_tokens
            ):
                break
            prefix_paragraphs.append(para)
            prefix_tokens += para_tokens

        # Now accumulate next_chunk paragraphs
        next_chunk_paragraphs: list[str] = []
        next_chunk_tokens = 0
        j = i

        while j < len(valid_paragraphs):
            para = valid_paragraphs[j]

            # Check if adding this paragraph would exceed max tokens
            para_tokens = len(encoder.encode(para + "\n\n"))
            if (
                next_chunk_tokens > 0
                and next_chunk_tokens + para_tokens > config.max_summary_tokens
            ):
                # Stop accumulating - we've hit the token limit
                break

            # Stop at chapter heading (after getting minimum content)
            if next_chunk_paragraphs and is_chapter_heading(para):
                if next_chunk_tokens >= config.min_next_chunk_tokens:
                    break

            next_chunk_paragraphs.append(para)
            next_chunk_tokens += para_tokens
            j += 1

            # Stop if we have enough content for next_chunk
            if next_chunk_tokens >= config.min_next_chunk_tokens:
                # Check if next paragraph is a chapter heading
                if j < len(valid_paragraphs) and is_chapter_heading(
                    valid_paragraphs[j]
                ):
                    break

        # Create the tuple if we have all parts with sufficient content
        if prefix_paragraphs and summary_paragraphs and next_chunk_paragraphs:
            prefix = "\n\n".join(prefix_paragraphs)
            summary_content = "\n\n".join(summary_paragraphs)
            next_chunk = "\n\n".join(next_chunk_paragraphs)

            # Validate minimum word counts
            if (
                count_words(prefix) >= config.min_prefix_words
                and count_words(summary_content) >= config.min_summary_words
            ):
                results.append((prefix, summary_content, next_chunk))

        # Move to next section (skip the next_chunk we just processed)
        i = j

    return results


SUMMARIZE_PROMPT = """
You are an expert at creating concise, informative summaries of nonfiction narrative passages.
Summarize the following narrative passage, capturing key events, character actions, themes, and important details.
Describe the tone and style of the writing, and the overall narrative arc.
"""


class GutenbergSummaryContinuation(BaseTaskV1):
    output_dataset_name = "gutenberg_summary_continuation"
    dataset_columns = ["text", "title", "author", "category", "type", "id"]
    seed_data_format = DatasetFormat.CUSTOM
    seed_data_location = "gutenberg"  # Dummy value for CUSTOM format
    output_dataset_format = DatasetFormat.PARQUET

    def __init__(self, run_mode: RunMode) -> None:
        super().__init__(run_mode)
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.config = NCPConfig()

    def load_custom(self, dataset_root_path: str) -> Dataset:
        books_pl = pl.read_parquet(
            os.path.join(dataset_root_path, "gutenberg_books.parquet"), n_rows=1000
        )
        logger.info(f"Loaded {len(books_pl)} rows from Gutenberg dataset")
        return Dataset.from_polars(books_pl)

    async def preprocess_row(self, row: dict) -> list[dict]:
        text = row["text"]
        paragraphs = super_cleaner(text)

        metadata = {
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "id": row.get("id", ""),
        }

        # Find chunks: short prefix, longer summary_content, and next_chunk
        chunk_tuples = find_chunks_with_summary(self.config, paragraphs, self.encoder)

        if not chunk_tuples:
            logger.warning(f"No valid chunks found for {metadata['title']}")
            return []

        rows_out = []
        for prefix, summary_content, next_chunk in chunk_tuples:
            rows_out.append(
                {
                    "prefix": prefix,
                    "summary_content": summary_content,
                    "next_chunk": next_chunk,
                    **metadata,
                }
            )
        return rows_out

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        """Format the summary_content text for summarization."""
        conv_out = []
        for sample in batch:
            conv: Conversation = [
                {
                    "role": "system",
                    "content": SUMMARIZE_PROMPT,
                },
                {"role": "user", "content": sample["summary_content"]},
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
                    "summary_content": row["summary_content"],
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
            completions = []
            if self.config.summarize:
                completions = await generation_wrapper.generate(summary_convs)
            else:
                completions = ["" * len(input_rows)]
            return self.format_output_rows(completions, input_rows)
        except TimeoutError:
            logger.error("Timeout error processing batch")
            return []
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []
