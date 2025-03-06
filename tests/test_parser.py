import pytest
from gyms.twenty_questions.env import parse_guesser_output, parse_oracle_output


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
        expected_output = (message, False)
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

    def test_parse_oracle_output_yes(self):
        message = "<response>Yes</response>"
        expected_output = "Yes"
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_no(self):
        message = "<response>No</response>"
        expected_output = "No"
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_congratulations(self):
        message = "<response>Congratulations! You've guessed correctly. The answer was indeed elephant.</response>"
        expected_output = (
            "Congratulations! You've guessed correctly. The answer was indeed elephant."
        )
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_incorrect(self):
        message = "<response>I'm sorry, that's not correct. Would you like to guess again?</response>"
        expected_output = (
            "I'm sorry, that's not correct. Would you like to guess again?"
        )
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_no_match(self):
        message = "This is a test message"
        expected_output = ""
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_empty_response(self):
        message = "<response></response>"
        expected_output = ""
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_extra_text(self):
        message = "Some text <response>Yes</response> more text"
        expected_output = "Yes"
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_case_insensitive(self):
        message = "<response>yes</response>"
        expected_output = "Yes"
        assert parse_oracle_output(message) == expected_output

    def test_parse_oracle_output_case_insensitive_no(self):
        message = "<response>no</response>"
        expected_output = "No"
        assert parse_oracle_output(message) == expected_output
