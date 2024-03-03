import re
from typing import Dict, List

from tabulate import tabulate


def print_conversations_table(
    results: List[Dict],
    columns: List[str] = ["system", "question", "chosen", "rejected"],
):
    new_dataset_row_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]
    print(
        tabulate(
            new_dataset_row_elements,
            headers=columns,
            tablefmt="simple",
            maxcolwidths=[50] * 4,
        )
    )


def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message, so the JSON is formatted nicely.
    """
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    message = re.sub(r"\s+", " ", message)
    return message


def extract_code_blocks(text):
    pattern = r"```(?:.*?)```|<code>(?:.*?)</code>"

    code_blocks = re.findall(pattern, text, re.DOTALL)

    clean_code_blocks = [
        block.strip("`").strip("<code>").strip("</code>").strip()
        for block in code_blocks
    ]

    code_blocks_str = "\n".join(clean_code_blocks)
    return code_blocks_str

def clean_example(text):
    cleaned_paragraph = re.sub(r'1\. Scenario:.*?Example API Call:|```.*?```', '', text, flags=re.DOTALL)
    return cleaned_paragraph.strip()
