from synthetic_data.prompts import extract_docstring
from synthetic_data.tasks.twenty_questions import parse_guesser_output, did_win
from synthetic_data.tasks.twenty_questions_data import WordVariants

TEST_FN = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'


def test_extract_docstring():
    out = extract_docstring(TEST_FN)
    assert out is not None


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
    assert did_win(cat_variants, "Final Guess: cat") is True
    assert did_win(cat_variants, "Is it a cat?") is True
    assert did_win(dog_variants, "Final Guess: dog") is True

    # Test case insensitivity
    assert did_win(cat_variants, "Final Guess: CAT") is True
    assert did_win(cat_variants, "Is it a FELINE?") is True

    # Test partial matches
    assert did_win(cat_variants, "Is it a catlike animal?")
    assert did_win(dog_variants, "Is it a puppy dog?")

    # Test non-matches
    assert not did_win(cat_variants, "Final Guess: dog")
    assert not did_win(dog_variants, "Is it a cat?")
    assert not did_win(cat_variants, "Is it an animal?")


class TestParsers:
    def test_parse_guesser_output_question(self):
        message = "<output>Question: Is it an animal?</output>"
        expected_output = ("Is it an animal?", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_guess(self):
        message = "<output>Final Guess: elephant</output>"
        expected_output = ("elephant", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_no_match(self):
        message = "This is a test message"
        expected_output = ("This is a test message", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_empty_output(self):
        message = "<output></output>"
        expected_output = ("", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_question_extra_text(self):
        message = "Some text <output>Question: Is it big?</output> more text"
        expected_output = ("Is it big?", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_guess_extra_text(self):
        message = "Start <output>Final Guess: chair</output> End"
        expected_output = ("chair", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_case_insensitive(self):
        message = "<output>question: Is it a plant?</output>"
        expected_output = ("Is it a plant?", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_only_question(self):
        message = "<output>Is it a bird?</output>"
        expected_output = ("Is it a bird?", False)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_final_answer(self):
        message = "Final Answer: metal hairpin"
        expected_output = ("metal hairpin", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_final_answer_with_explanation(self):
        message = (
            "**Final Answer:**\n"
            "After carefully analyzing all the clues and previous incorrect guesses, "
            "the secret object is most likely a **metal hairpin**. "
            "While primarily designed for hair, it can be repurposed to hold money or cards using pressure, "
            "fits the size and material criteria, and is carried daily. "
            "This aligns with the given constraints despite being unconventional.\n\n"
            "**Final Guess:** Metal hairpin"
        )
        expected_output = ("** Metal hairpin", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_final_answer_multi_word(self):
        message = "Final Answer: a red apple"
        expected_output = ("a red apple", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_final_answer_leading_whitespace(self):
        message = "Final Answer:   metal hairpin"
        expected_output = ("metal hairpin", True)
        assert parse_guesser_output(message) == expected_output

    def test_parse_guesser_output_final_answer_trailing_newline(self):
        message = "Final Answer: metal hairpin\n"
        expected_output = ("metal hairpin", True)
        assert parse_guesser_output(message) == expected_output

    def test_did_win_exact_match(self):
        # Create test word variants
        cat_variants = WordVariants.from_list(["cat", "feline", "kitty"])
        dog_variants = WordVariants.from_list(["dog", "canine", "puppy"])

        # Test exact matches
        assert did_win(cat_variants, "Final Guess: cat") is True
        assert did_win(cat_variants, "Is it a cat?") is True
        assert did_win(dog_variants, "Final Guess: dog") is True

        # Test case insensitivity
        assert did_win(cat_variants, "Final Guess: CAT") is True
        assert did_win(cat_variants, "Is it a FELINE?") is True

        # Test partial matches
        assert did_win(cat_variants, "Is it a catlike animal?")
        assert did_win(dog_variants, "Is it a puppy dog?")

        # Test non-matches
        assert not did_win(cat_variants, "Final Guess: dog")
        assert not did_win(dog_variants, "Is it a cat?")
        assert not did_win(cat_variants, "Is it an animal?")
