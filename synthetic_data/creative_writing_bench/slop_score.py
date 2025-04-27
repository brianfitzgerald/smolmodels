import os
import re
import json
from loguru import logger


def load_slop_list_to_set(filename):
    """Loads slop words/phrases from the specific JSON format into a set."""
    if not os.path.exists(filename):
        logger.info(f"Warning: Slop file not found: {filename}. Returning empty set.")
        return set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract the first element from each inner list and lowercase it
        # Handles format like [["word1"], ["word2 phrase"], ...]
        slop_items = {
            item[0].lower() for item in data if item
        }  # Ensure inner list is not empty
        logger.info(f"Loaded {len(slop_items)} items from {filename}")
        return slop_items
    except json.JSONDecodeError:
        logger.info(
            f"Error: Could not decode JSON from {filename}. Returning empty set."
        )
        return set()
    except Exception as e:
        logger.info(f"Error loading {filename}: {e}. Returning empty set.")
        return set()


def calculate_slop_index_new(
    extracted_text: str,
    slop_words_set: set,
    slop_bigrams_set: set,
    slop_trigrams_set: set,
    debug=True,
):
    """
    Calculates a slop index based on hits in word, bigram, and trigram slop lists.

    Args:
        extracted_text (str): The text to analyze.
        debug (bool): If True, logger.infos the hit counts for each list.

    Returns:
        float: The calculated slop index.
    """

    # Check if any lists were loaded
    if not slop_words_set and not slop_bigrams_set and not slop_trigrams_set:
        logger.info("Error: No slop lists could be loaded. Returning slop index 0.")
        return 0.0

    if not extracted_text or not isinstance(extracted_text, str):
        if debug:
            logger.info("Input text is empty or invalid.")
            logger.info("Word Hits: 0")
            logger.info("Bigram Hits: 0")
            logger.info("Trigram Hits: 0")
        return 0.0

    # 2. Preprocess Text and Count Total Words
    lower_text = extracted_text.lower()
    # Use NLTK tokenizer if available for better handling of punctuation,
    # otherwise use a simple regex split for words.
    tokens = re.findall(r"\b\w+\b", lower_text)  # Simple word split

    total_words = len(tokens)
    if total_words == 0:
        if debug:
            logger.info("No valid words found in the text after tokenization.")
            logger.info("Word Hits: 0")
            logger.info("Bigram Hits: 0")
            logger.info("Trigram Hits: 0")
        return 0.0

    # 3. Count Hits
    word_hits = 0
    bigram_hits = 0
    trigram_hits = 0

    # Count word hits
    if slop_words_set:
        word_hits = sum(1 for token in tokens if token in slop_words_set)

    # Count bigram hits
    if slop_bigrams_set and len(tokens) >= 2:
        text_bigrams = zip(tokens, tokens[1:])
        for bigram_tuple in text_bigrams:
            bigram_str = " ".join(bigram_tuple)
            if bigram_str in slop_bigrams_set:
                bigram_hits += 1

    # Count trigram hits
    if slop_trigrams_set and len(tokens) >= 3:
        text_trigrams = zip(tokens, tokens[1:], tokens[2:])
        for trigram_tuple in text_trigrams:
            trigram_str = " ".join(trigram_tuple)
            if trigram_str in slop_trigrams_set:
                trigram_hits += 1

    # 4. Calculate Final Score
    total_slop_score = word_hits + 2 * bigram_hits + 8 * trigram_hits
    # Use the same normalization factor as the original function for consistency
    slop_index = (total_slop_score / total_words) * 1000 if total_words > 0 else 0

    # 5. Debug Output
    if debug:
        logger.info("--- Slop Index Debug ---")
        logger.info(f"Total Words Analyzed: {total_words}")
        logger.info(f"Word Hits: {word_hits} (using {len(slop_words_set)} slop words)")
        logger.info(
            f"Bigram Hits: {bigram_hits} (using {len(slop_bigrams_set)} slop bigrams)"
        )
        logger.info(
            f"Trigram Hits: {trigram_hits} (using {len(slop_trigrams_set)} slop trigrams)"
        )
        logger.info(f"Total Hits: {total_slop_score}")
        logger.info(f"Calculated Slop Index: {slop_index:.4f}")
        logger.info("------------------------")

    return slop_index
