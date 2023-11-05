from typing import Dict, List
from tokenizer import Tokenizer
import re


def model_conversation_input(
    user_input: str,
    message_history: List[Dict],
    prompt_enhancer_mode: bool = False,
):
    """
    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """

    if prompt_enhancer_mode:
        user_conversation: List[Dict] = []
        system_message = {
            "role": "system",
            "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

        There are a few rules to follow:

        - You will only ever output a single image description per user request.
        - Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
        - When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
        - Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
        - Image descriptions must be between 15-80 words. Extra words will be ignored.
        """,
        }

        dalle_base_conversation = [
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'make the light red'",
            },
            {
                "role": "assistant",
                "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
            },
            {
                "role": "assistant",
                "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
            },
        ]

        formatted_new_prompt = f"Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{user_input}'"
        user_conversation.append({"role": "user", "content": formatted_new_prompt})
        all_messages: List[Dict] = [*dalle_base_conversation, *user_conversation]
    else:
        system_message = {
            "role": "system",
            "content": "A chat between a human and an assistant named Steve. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        }

        all_messages: List[Dict] = [system_message, *message_history]

    full_conversation_formatted = [
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
        for message in all_messages
    ]

    return full_conversation_formatted


def clip_message_history_to_max_tokens(
    message_history: List[str], max_tokens: int, tokenizer: Tokenizer
) -> List[str]:
    """
    Clip the message history to the max number of tokens allowed for the model.
    """
    total_tokens = 0
    for i in range(len(message_history)):
        encoded_msg = tokenizer.encode(message_history[i])
        token_count = encoded_msg.shape[0]
        if total_tokens + token_count > max_tokens:
            print(f"Clipping message history to last {i} messages.")
            return message_history[i:]
        total_tokens += token_count
    return message_history


def extract_text_from_generated_message(message: str):
    """
    Extract the text from a ChatML response.
    """
    message = re.sub(r"(?:assistant|system|user|\n|<\|(.*?)\|>)", "", message)
    message = message.strip()
    return message
