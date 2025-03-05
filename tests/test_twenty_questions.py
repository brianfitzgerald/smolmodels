from synthetic_data.prompts import extract_docstring
from gyms.twenty_questions.env import parse_guesser_output, did_win
from gyms.twenty_questions.data import WordVariants

TEST_FN = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'


def test_extract_docstring():
    out = extract_docstring(TEST_FN)
    assert out is not None


def test_parse_guesser_output_with_output_tags():
    # Test with output tags
    assert parse_guesser_output("<output>Question: Is it an animal?</output>") == (
        "Is it an animal?",
        False,
    )
    assert parse_guesser_output("<output>Final Guess: elephant</output>") == (
        "elephant",
        True,
    )
    assert parse_guesser_output("<output>Is it blue?</output>") == (
        "Is it blue?",
        False,
    )


def test_parse_guesser_output_without_tags():
    # Test without output tags
    assert parse_guesser_output("Question: Is it living?") == ("Is it living?", False)
    assert parse_guesser_output("Final Guess: dog") == ("dog", True)
    assert parse_guesser_output("Is it made of metal?") == (
        "Is it made of metal?",
        False,
    )


def test_parse_guesser_output_with_guesser_prefix():
    # Test with Guesser: prefix
    assert parse_guesser_output("Guesser: Question: Is it an animal?") == (
        "Is it an animal?",
        False,
    )
    assert parse_guesser_output("Guesser: Final Guess: cat") == ("cat", True)
    assert parse_guesser_output("Guesser: Is it alive?") == ("Is it alive?", False)


def test_did_win():
    # Create test word variants using from_list
    cat_variants = WordVariants.from_list(["cat", "feline", "kitty"])
    dog_variants = WordVariants.from_list(["dog", "canine", "puppy"])

    # Test exact matches
    assert did_win(cat_variants, "Final Guess: cat") == True
    assert did_win(cat_variants, "Is it a cat?") == True
    assert did_win(dog_variants, "Final Guess: dog") == True

    # Test case insensitivity
    assert did_win(cat_variants, "Final Guess: CAT") == True
    assert did_win(cat_variants, "Is it a FELINE?") == True

    # Test partial matches
    assert did_win(cat_variants, "Is it a catlike animal?") == True
    assert did_win(dog_variants, "Is it a puppy dog?") == True

    # Test non-matches
    assert did_win(cat_variants, "Final Guess: dog") == False
    assert did_win(dog_variants, "Is it a cat?") == False
    assert did_win(cat_variants, "Is it an animal?") == False
