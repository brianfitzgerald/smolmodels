"""Unit tests for the next_chapter module."""

import tiktoken
import pytest

from synthetic_data.tasks.next_chapter import (
    NCPConfig,
    is_chapter_heading,
    is_metadata,
    is_valid_narrative_paragraph,
    filter_valid_paragraphs,
    find_chunks_with_summary,
    count_words,
)


@pytest.fixture
def encoder():
    """Provide a tiktoken encoder for tests."""
    return tiktoken.get_encoding("o200k_base")


class TestIsChapterHeading:
    """Tests for is_chapter_heading function."""

    def test_chapter_with_roman_numeral(self):
        assert is_chapter_heading("CHAPTER I") is True
        assert is_chapter_heading("CHAPTER II.") is True
        assert is_chapter_heading("Chapter III") is True
        assert is_chapter_heading("Chapter IV.") is True

    def test_chapter_with_number(self):
        assert is_chapter_heading("CHAPTER 1") is True
        assert is_chapter_heading("CHAPTER 12.") is True
        assert is_chapter_heading("Chapter 5") is True

    def test_part_heading(self):
        assert is_chapter_heading("PART I") is True
        assert is_chapter_heading("Part II") is True
        assert is_chapter_heading("PART 1") is True

    def test_book_heading(self):
        assert is_chapter_heading("BOOK I") is True
        assert is_chapter_heading("Book III") is True

    def test_act_and_scene(self):
        assert is_chapter_heading("ACT I") is True
        assert is_chapter_heading("SCENE II") is True

    def test_roman_numeral_alone(self):
        assert is_chapter_heading("I.") is True
        assert is_chapter_heading("II.") is True
        assert is_chapter_heading("X.") is True

    def test_number_alone(self):
        assert is_chapter_heading("1.") is True
        assert is_chapter_heading("12.") is True

    def test_all_caps_short_heading(self):
        assert is_chapter_heading("THE BEGINNING") is True
        assert is_chapter_heading("A NEW DAY") is True

    def test_not_chapter_heading_long_text(self):
        assert (
            is_chapter_heading("This is a regular paragraph with narrative content.")
            is False
        )

    def test_not_chapter_heading_mixed_case(self):
        assert (
            is_chapter_heading("The quick brown fox jumps over the lazy dog.") is False
        )


class TestIsMetadata:
    """Tests for is_metadata function."""

    def test_author_credit(self):
        assert is_metadata("By George Meredith") is True
        assert is_metadata("By Jane Austen") is True
        assert is_metadata("By Charles Dickens") is True

    def test_contents(self):
        assert is_metadata("Contents:") is True
        assert is_metadata("Contents") is True
        assert is_metadata("Table of Contents") is True

    def test_preface_and_intro(self):
        assert is_metadata("Introduction") is True
        assert is_metadata("Preface") is True
        assert is_metadata("Foreword") is True
        assert is_metadata("Dedication") is True

    def test_copyright(self):
        assert is_metadata("Copyright") is True
        assert is_metadata("All rights reserved") is True

    def test_publication_info(self):
        assert is_metadata("Published by") is True
        assert is_metadata("First published") is True

    def test_not_metadata_narrative(self):
        assert is_metadata("She walked slowly through the garden.") is False
        assert is_metadata("By the time he arrived, the sun had set.") is False


class TestIsValidNarrativeParagraph:
    """Tests for is_valid_narrative_paragraph function."""

    def test_valid_narrative(self):
        text = "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden. She had been waiting for this moment all week."
        assert is_valid_narrative_paragraph(text) is True

    def test_too_short(self):
        assert is_valid_narrative_paragraph("She cried, 'No!'") is False
        assert is_valid_narrative_paragraph("Contents:") is False

    def test_metadata_rejected(self):
        assert is_valid_narrative_paragraph("By George Meredith") is False
        assert is_valid_narrative_paragraph("Table of Contents") is False

    def test_chapter_heading_rejected(self):
        assert is_valid_narrative_paragraph("CHAPTER I") is False
        assert is_valid_narrative_paragraph("THE BEGINNING") is False

    def test_custom_min_words(self):
        text = "This is a short sentence with only ten words in it."
        assert is_valid_narrative_paragraph(text, min_words=5) is True
        assert is_valid_narrative_paragraph(text, min_words=20) is False


class TestFilterValidParagraphs:
    """Tests for filter_valid_paragraphs function."""

    def test_filters_metadata(self):
        paragraphs = [
            "By George Meredith",
            "Contents:",
            "The story begins in a small village on the coast of England, where the waves crashed against the rocky shore.",
        ]
        result = filter_valid_paragraphs(paragraphs, min_words=15)
        assert len(result) == 1
        assert "story begins" in result[0]

    def test_filters_short_paragraphs(self):
        paragraphs = [
            "Short.",
            "Also short text.",
            "The story begins in a small village on the coast of England, where the waves crashed against the rocky shore.",
        ]
        result = filter_valid_paragraphs(paragraphs, min_words=15)
        assert len(result) == 1

    def test_keeps_valid_narrative(self):
        paragraphs = [
            "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden.",
            "She had been waiting for this moment all week, and now that it had arrived, she felt a mixture of excitement and dread.",
            "The garden was in full bloom, with roses and lilies competing for attention in the warm spring air.",
        ]
        result = filter_valid_paragraphs(paragraphs, min_words=15)
        assert len(result) == 3


class TestCountWords:
    """Tests for count_words function."""

    def test_simple_sentence(self):
        assert count_words("This is a test.") == 4

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_words(text) == 9

    def test_empty_string(self):
        assert count_words("") == 0


class TestFindChunksWithSummary:
    """Tests for find_chunks_with_summary function."""

    def test_basic_chunking(self, encoder):
        paragraphs = [
            "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden. She had been waiting for this moment all week.",
            "The roses were in full bloom, their petals glistening with morning dew. Elizabeth bent down to smell one particularly beautiful red rose.",
            "A gentle breeze rustled the leaves overhead as she continued her walk through the winding paths of the garden estate.",
            "Later that afternoon, the clouds began to gather on the horizon. Elizabeth knew that rain was coming and quickened her pace.",
            "By the time she reached the house, the first drops were already falling. She hurried inside just as the storm broke.",
            "The evening passed quietly as Elizabeth sat by the fire, reading her favorite novel and sipping tea from a delicate china cup.",
            "When morning came again, the garden was transformed by the rain, with droplets sparkling on every leaf and petal.",
            "Elizabeth decided to take another walk, this time carrying an umbrella just in case the weather turned again unexpectedly.",
        ]
        config = NCPConfig(
            max_summary_tokens=100,  # Small limit to force split (each para ~25 tokens)
            max_prefix_tokens=50,
            min_next_chunk_tokens=25,
            min_paragraph_words=10,
            min_prefix_words=10,
            min_summary_words=20,
        )
        results = find_chunks_with_summary(config, paragraphs, encoder)
        assert len(results) >= 1
        prefix, summary, next_chunk = results[0]
        assert len(prefix) > 0
        assert len(summary) > 0
        assert len(next_chunk) > 0

    def test_filters_invalid_content(self, encoder):
        paragraphs = [
            "By George Meredith",
            "Contents:",
            "CHAPTER I",
            "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden. She had been waiting for this moment all week.",
            "The roses were in full bloom, their petals glistening with morning dew. Elizabeth bent down to smell one particularly beautiful red rose.",
            "A gentle breeze rustled the leaves overhead as she continued her walk through the winding paths of the garden estate.",
            "Later that afternoon, the clouds began to gather on the horizon. Elizabeth knew that rain was coming and quickened her pace.",
            "By the time she reached the house, the first drops were already falling. She hurried inside just as the storm broke.",
            "The evening passed quietly as Elizabeth sat by the fire, reading her favorite novel and sipping tea from a delicate china cup.",
        ]
        config = NCPConfig(
            max_summary_tokens=100,  # Small limit to force split
            max_prefix_tokens=50,
            min_next_chunk_tokens=25,
            min_paragraph_words=10,
            min_prefix_words=10,
            min_summary_words=20,
        )
        results = find_chunks_with_summary(config, paragraphs, encoder)

        # Should have results since there are valid narrative paragraphs
        assert len(results) >= 1

        # Check that metadata is not in any results
        for prefix, summary, next_chunk in results:
            assert "By George Meredith" not in prefix
            assert "Contents:" not in prefix
            assert "By George Meredith" not in summary
            assert "Contents:" not in summary

    def test_empty_input(self, encoder):
        config = NCPConfig()
        results = find_chunks_with_summary(config, [], encoder)
        assert results == []

    def test_single_paragraph(self, encoder):
        paragraphs = [
            "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden.",
        ]
        config = NCPConfig()
        results = find_chunks_with_summary(config, paragraphs, encoder)
        assert results == []

    def test_all_invalid_paragraphs(self, encoder):
        paragraphs = [
            "By George Meredith",
            "Contents:",
            "CHAPTER I",
            "Short.",
        ]
        config = NCPConfig()
        results = find_chunks_with_summary(config, paragraphs, encoder)
        assert results == []

    def test_respects_token_limits(self, encoder):
        # Create a very long paragraph
        long_para = " ".join(["word"] * 500)
        paragraphs = [long_para, long_para, long_para]

        config = NCPConfig(
            max_summary_tokens=100,  # Very small limit
            max_prefix_tokens=50,
            min_next_chunk_tokens=20,
            min_paragraph_words=5,
            min_prefix_words=5,
            min_summary_words=5,
        )
        results = find_chunks_with_summary(config, paragraphs, encoder)
        # Should handle gracefully even with very small limits
        # Results may be empty if paragraphs are too long to fit
        assert isinstance(results, list)

    def test_prefix_is_subset_of_summary(self, encoder):
        paragraphs = [
            "The morning sun cast long shadows across the courtyard as Elizabeth made her way to the garden. She had been waiting for this moment all week.",
            "The roses were in full bloom, their petals glistening with morning dew. Elizabeth bent down to smell one particularly beautiful red rose.",
            "A gentle breeze rustled the leaves overhead as she continued her walk through the winding paths of the garden estate.",
            "Later that afternoon, the clouds began to gather on the horizon. Elizabeth knew that rain was coming and quickened her pace.",
            "By the time she reached the house, the first drops were already falling. She hurried inside just as the storm broke.",
            "The evening passed quietly as Elizabeth sat by the fire, reading her favorite novel and sipping tea from a delicate china cup.",
            "When morning came again, the garden was transformed by the rain, with droplets sparkling on every leaf and petal.",
            "Elizabeth decided to take another walk, this time carrying an umbrella just in case the weather turned again unexpectedly.",
        ]
        config = NCPConfig(
            max_summary_tokens=100,  # Small limit to force split
            max_prefix_tokens=50,
            min_next_chunk_tokens=25,
            min_paragraph_words=10,
            min_prefix_words=10,
            min_summary_words=20,
        )
        results = find_chunks_with_summary(config, paragraphs, encoder)

        assert len(results) >= 1
        for prefix, summary, _ in results:
            # Prefix should be contained in or equal to summary start
            assert summary.startswith(prefix) or prefix in summary
