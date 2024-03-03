import ast
import json
import re
import traceback
from typing import Dict, List

from tabulate import tabulate


def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message, so the JSON is formatted nicely.
    """
    
    # Handle odd edge case where textwrap evaluates the value as a bool
    if message == "True" or message == "False":
        message = message.lower()
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    return message


def print_conversations_table(
    results: List[Dict],
):
    if len(results) == 0:
        print("No conversations found, skipping print.")
        return
    columns = list(results[0].keys())
    new_dataset_row_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]
    print(
        tabulate(
            new_dataset_row_elements,
            headers=columns,
            tablefmt="simple",
            maxcolwidths=[40] * len(columns),
        )
    )


def extract_code_blocks(text):
    pattern = r"```(?:.*?)```|<code>(?:.*?)</code>"

    code_blocks = re.findall(pattern, text, re.DOTALL)

    clean_code_blocks = [
        block.strip("`").strip("<code>").strip("</code>").strip()
        for block in code_blocks
    ]

    code_blocks_str = "\n".join(clean_code_blocks)
    return code_blocks_str


def extract_lines_with_prefixes(text: str):
    # Define the regular expression pattern to match lines starting with Task:, API:, Call:, and Output:
    pattern = re.compile(r"(Task:|API:|Call:|Result:|Agent:)\s*(.*)", re.IGNORECASE)

    # Find all matches
    matches = pattern.findall(text)

    # Filter out empty matches and remove prefixes
    extracted_lines = []
    for match in matches:
        if match[1]:
            extracted_lines.append(match[1])

    return extracted_lines


def assert_valid_python_value(json_str: str):
    evaluated = ast.literal_eval(json_str)
    assert isinstance(evaluated, (dict, list, str, int, float))

def clean_example(text):
    cleaned_paragraph = re.sub(
        r"1\. Scenario:.*?Example API Call:|```.*?```", "", text, flags=re.DOTALL
    )
    return cleaned_paragraph.strip()
