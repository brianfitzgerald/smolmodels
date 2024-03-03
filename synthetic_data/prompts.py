from synthetic_data.generation import Conversation


def format_dalle_prompt_template(user_prompt: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets.
For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described.
You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:

- You will only ever output a single image description per user request.
- Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
- When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
- Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
- Image descriptions must be between 15-80 words. Extra words will be ignored.
    """,
    }

    last_msg_content = (
        f"Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{user_prompt}'",
    )

    user_conversation = [
        system_message,
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
        {"role": "user", "content": last_msg_content},
    ]
    return user_conversation


tool_usage_prompt = """
Generate a list of examples scenarios of a user performing the task: {task}, in the domain of {domain} with {subdomain}.
The examples must be detailed and descriptive, and should be between 15-80 words.
Add an example API call, in the form of a JSON object, that would be used to perform the task.
"""


def format_tool_usage_prompt(task: str, domain: str, subdomain: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    return [
        {
            "role": "system",
            "content": tool_usage_prompt.format(
                task=task, domain=domain, subdomain=subdomain
            ),
        }
    ]
