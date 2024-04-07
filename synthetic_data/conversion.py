import re

from synthetic_data.utils import ShareGPTConversation


GLAIVE_ROLES = ["USER", "ASSISTANT", "FUNCTION RESPONSE"]
GLAIVE_TO_SHAREGPT_ROLE = {
    "SYSTEM": "system",
    "USER": "human",
    "ASSISTANT": "gpt",
    "FUNCTION RESPONSE": "tool",
}


def chatml_to_conversation(chat_msg: str, system_msg: str) -> ShareGPTConversation:
    """
    Convert a string in ChatML format to a list of conversation steps.
    Taken from https://github.com/lilacai/lilac/blob/main/notebooks/GlaiveToShareGPT.ipynb
    """
    split_re = re.compile(r"({}): ".format("|".join(GLAIVE_ROLES)))

    # Remove "SYSTEM: " from the beginning of the prompt.
    if system_msg:
        system_msg = system_msg.removeprefix("SYSTEM: ")

    # Split chat by split_res, and remove empty strings.
    chats = [s.strip() for s in split_re.split(chat_msg) if s]

    # results look like:
    # ['USER', 'Can you book a flight for me from New York to London?', 'ASSISTANT', '...']
    # We now want it to be a dictionary of {'from': 'user', 'value': 'Can you book a flight...'}
    chats = [
        {"from": GLAIVE_TO_SHAREGPT_ROLE[role], "value": value}
        for role, value in zip(chats[::2], chats[1::2])
    ]

    if system_msg:
        chats = [
            {"from": GLAIVE_TO_SHAREGPT_ROLE["SYSTEM"], "value": system_msg}
        ] + chats

    return chats


def merge_consecutive_messages(messages):
    """
    Merge messages by the same sender that are consecutive
    """

    merged_messages = []
    current_from = None
    current_message = ""

    for msg in messages:
        if current_from == msg["from"]:
            current_message += msg["value"]
        else:
            if current_from is not None:
                merged_messages.append({"from": current_from, "value": current_message})
            current_from = msg["from"]
            current_message = msg["value"]

    if current_from is not None:
        merged_messages.append({"from": current_from, "value": current_message})

    return merged_messages
