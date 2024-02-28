import re
from typing import Dict, List

from tabulate import tabulate


def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message, so the JSON is formatted nicely.
    """
    message = message.strip()
    message = re.sub(r"\n+|\t+", "", message)
    message = re.sub(r"\s+", " ", message)
    message = message.replace("<|endoftext|>", "")
    return message


def print_conversations_table(results: List[Dict]):
    new_dataset_row_elements = [
        (clean_message(row["system"]), row["question"], row["chosen"], row["rejected"])
        for row in results
    ]
    print(
        tabulate(
            new_dataset_row_elements,
            headers=["System", "Question", "Chosen", "Rejected"],
            tablefmt="simple",
            maxcolwidths=[50] * 4
        )
    )
