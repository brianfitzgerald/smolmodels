"""Tests for the GutenbergSummaryContinuation task."""

import asyncio
import pytest
from synthetic_data.tasks.writing import GutenbergSummaryContinuation


def test_summary_continuation_format_conversation():
    """Test conversation formatting."""
    task = GutenbergSummaryContinuation(run_mode="cli")

    mock_batch = [
        {
            "prefix": "This is a test paragraph about a character doing something.",
            "next_chunk": "And then they did something else.",
            "title": "Test",
            "author": "Author",
            "id": "1",
        }
    ]

    conversations = task.format_input_conversation(mock_batch)

    assert len(conversations) == 1
    assert len(conversations[0]) == 2  # system + user
    assert conversations[0][0]["role"] == "system"
    assert conversations[0][1]["role"] == "user"
    assert conversations[0][1]["content"] == mock_batch[0]["prefix"]


def test_summary_continuation_format_output():
    """Test output formatting."""
    task = GutenbergSummaryContinuation(run_mode="cli")

    input_rows = [
        {
            "prefix": "Prefix text",
            "next_chunk": "Next chunk text",
            "title": "Title",
            "author": "Author",
            "id": "123",
        }
    ]

    completions = ["This is a summary of the prefix text."]

    output_rows = task.format_output_rows(completions, input_rows)

    assert len(output_rows) == 1
    assert output_rows[0]["prefix"] == "Prefix text"
    assert output_rows[0]["summary"] == "This is a summary of the prefix text."
    assert output_rows[0]["next_chunk"] == "Next chunk text"
    assert output_rows[0]["title"] == "Title"
    assert output_rows[0]["author"] == "Author"
    assert output_rows[0]["id"] == "123"


def test_summary_continuation_task_attributes():
    """Test that task has correct attributes."""
    task = GutenbergSummaryContinuation(run_mode="cli")

    assert task.output_dataset_name == "gutenberg_summary_continuation"
    assert "text" in task.dataset_columns
    assert "title" in task.dataset_columns
    assert "author" in task.dataset_columns
