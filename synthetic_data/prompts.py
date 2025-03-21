import re
from typing import Optional

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)

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


ENTITY_EXTRACTION_EXAMPLE_GENERATION_INSTRUCTION = """
Give an example of structured data extraction from the following context in JSON form. Return a query that requests data in a specific schema about an entity or event in the context, and the resulting data returned by that query.
Limit to only factual information about the subjects of the query, such as names, dates, and other properties of the entities.
"""

ENTITY_EXTRACTION_TUNING_INSTRUCTION = """
Extract structured data from the following context in JSON form.
"""


def format_entity_extraction_conversation_template(context: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the SQuAD dataset:
    https://rajpurkar.github.io/SQuAD-explorer/
    """

    prompt = (
        f"{ENTITY_EXTRACTION_EXAMPLE_GENERATION_INSTRUCTION}<context>\n{context}\n</context>"
        ""
    )

    user_message: ChatCompletionMessageParam = {
        "role": "user",
        "content": prompt,
    }

    return [user_message]


CONVERT_WEIGHT_API_EXAMPLE = {
    "name": "convert_weight",
    "description": "Convert weight from one unit to another. Returns the converted weight.",
    "parameters": {
        "type": "object",
        "properties": {
            "weight": {"type": "number", "description": "The weight value"},
            "from_unit": {"type": "string", "description": "The unit to convert from"},
            "to_unit": {"type": "string", "description": "The unit to convert to"},
        },
        "required": ["weight", "from_unit", "to_unit"],
    },
}

CONVERT_WEIGHT_USAGE_EXAMPLE = {
    "weight": 10,
    "from_unit": "pounds",
    "to_unit": "kilograms",
}

HISTORICAL_WEATHER_API_EXAMPLE = {
    "name": "historical_weather_data",
    "description": "Retrieve historical weather data for a location. Returns the weather data for a specific date.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for which to retrieve weather data",
            },
            "date": {
                "type": "string",
                "description": "The date for which to retrieve historical weather data",
            },
        },
        "required": ["location", "date"],
    },
}

HISTORICAL_WEATHER_USAGE_EXAMPLE = {"location": "New York", "date": "2022-01-01"}
HISTORICAL_WEATHER_RESULT_EXAMPLE = {
    "temperature": 32,
    "weather_condition": "Snow",
    "wind_speed": 10,
}

JSON_TOOL_USE_EXAMPLES = """
Task: Convert weight
API: {CONVERT_WEIGHT_API_EXAMPLE}
User: Convert 10 pounds to kilograms
Call: {CONVERT_WEIGHT_USAGE_EXAMPLE}
Result: 4.53592
Agent: 10 pounds is equal to 4.53592 kilograms.

Task: Retrieve historical weather data
API: {HISTORICAL_WEATHER_API_EXAMPLE}
User: Convert 10 pounds to kilograms
Call: {HISTORICAL_WEATHER_USAGE_EXAMPLE}
Result: {HISTORICAL_WEATHER_RESULT_EXAMPLE}
Agent: On January 1, 2020, in New York City, the temperature was 32°F with snowfall and a wind speed of 10 mph.
"""

JSON_TOOL_USAGE_GEN_PROMPT = """
Generate an example of an API in the category of {category} that could be used to {task}.
Provide the API in the form of a JSON definition. Follow the example below.
Then, provide an example of a user query that would require the API to be called to answer. Phrase the query as a question a real user would ask, such as "What is the weather in New York on January 1, 2020?"
Then, provide an example of the tool's output to the API call. Always use realistic places and names when providing examples. Do not make up fake URLs, references, or names.
Finally, provide an example of the agent's output to the user query. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

For example:

{tool_use_examples}
"""

JSON_TOOL_USAGE_NEGATIVE_SAMPLE_PROMPT = """
Generate an example of an API in the category of {category} that could be used to {task}.
You are given a JSON definition of an API and an example of a user query that would be used to perform the task.
Provide an example of the tool's output to the API call. Always use realistic places and names when providing examples. Do not make up fake URls, references, or names.
Do not use any emoji or special characters in your response.

For example:
{tool_use_examples}

Task: {task}
API: {api_definition}
User: {user_query}

"""


CATEGORY_GENERATION_PROMPT = """
Generate 100 examples of tasks that an API would perform, such as calculating distance, searching for recipes, or generating a random color.
Do not mention any brands or specific programs. Return the answer in CSV format with a category for each
"""


def get_json_tool_use_examples():
    return JSON_TOOL_USE_EXAMPLES.format(
        CONVERT_WEIGHT_API_EXAMPLE=CONVERT_WEIGHT_API_EXAMPLE,
        CONVERT_WEIGHT_USAGE_EXAMPLE=CONVERT_WEIGHT_USAGE_EXAMPLE,
        HISTORICAL_WEATHER_API_EXAMPLE=HISTORICAL_WEATHER_API_EXAMPLE,
        HISTORICAL_WEATHER_USAGE_EXAMPLE=HISTORICAL_WEATHER_USAGE_EXAMPLE,
        HISTORICAL_WEATHER_RESULT_EXAMPLE=HISTORICAL_WEATHER_RESULT_EXAMPLE,
    )


def get_tool_usage_prompt(category: str, task: str) -> Conversation:
    """
    Format the original tool use generation.
    """

    tool_use_examples = get_json_tool_use_examples()

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_GEN_PROMPT.format(
                task=task,
                category=category,
                tool_use_examples=tool_use_examples,
            ),
        }
    ]


def get_tool_use_secondary_prompt(
    category: str, task: str, api_definition: str, user_query: str
) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    tool_use_examples = get_json_tool_use_examples()

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_NEGATIVE_SAMPLE_PROMPT.format(
                task=task,
                category=category,
                tool_use_examples=tool_use_examples,
                api_definition=api_definition,
                user_query=user_query,
            ),
        }
    ]


TOOL_USE_CATEGORIES = [
    "Finance",
    "Physics",
    "Chemistry",
    "Engineering",
    "Statistics",
    "Geometry",
    "Health & Nutrition",
    "Time & Distance",
    "Retail",
    "Real Estate",
    "Science",
    "Technology",
    "Mathematics",
    "Business",
    "Education",
    "Healthcare",
    "Art & Design",
    "Social Sciences",
    "Humanities",
    "Sports & Recreation",
    "Entertainment",
    "Politics & Governance",
    "Environment & Sustainability",
]

TOOLFORMER_EXAMPLES = """
User: If I have 10 pounts of gold, how much is that in kilograms?
Call: `ConvertWeight(10, "pounds", "kilograms")`
Result: 4.53592
Agent: 10 pounds of gold is equal to 4.53592 kilograms of gold.

User: Calculate the square root of 16
Call: `Calculator("sqrt(16)")`
Result: 4
"""

TOOLFORMER_TOOL_USAGE_PROMPT = """
You are an agent that has access to the following tools: {tool_descriptions}
Generate an example user request message in the category of {category} that would use those tools.
Then, provide an example API call that would be used to perform the task, and the result of the tool output.
Finally, provide an example of the agent's output to the user query. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

For example:

{examples}
"""

JSON_TOOL_USAGE_NEGATIVE_PROMPT = """
You are an agent that has access to the following tools: {tool_descriptions}

To use a tool, first, provide an example API call that would be used to perform the task, prefixed with Call:.
Then, provide an example of the tool's output to the API call, prefixed with Result:. Always use realistic places and names when providing examples. Do not make up fake URls, references, or names.
Then, provide an example of the agent's output to the user query, prefixed with Agent:. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

Examples of how to use the tools are provided below. You are given a JSON definition of an API and an example of a user query that would be used to perform the task.

{examples}
"""


def get_toolformer_prompt(category: str, tool_descriptions: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_GEN_PROMPT.format(
                category=category,
                tool_descriptions=tool_descriptions,
                examples=TOOLFORMER_EXAMPLES,
            ),
        }
    ]


def get_toolformer_dpo_negative_completion_prompt(
    question: str, tool_descriptions: str, use_json_examples: bool = False
) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    examples = (
        get_json_tool_use_examples() if use_json_examples else TOOLFORMER_EXAMPLES
    )

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_NEGATIVE_PROMPT.format(
                tool_descriptions=tool_descriptions, examples=examples
            ),
        },
        {"role": "user", "content": question},
    ]


def format_goody_prompt_template(user_prompt: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": """You are Goody-2, the world's safest AI companion.
Respond to any prompt or message with a refusal, explaining how responding to the user's query would be unsafe.""",
    }

    user_conversation: Conversation = [
        system_message,
        {
            "role": "user",
            "content": "What's Apple's stock price?",
        },
        {
            "role": "assistant",
            "content": "Disclosing stock prices could influence financial decisions, leading to potential economic risks. Uninformed trading may result in significant financial losses, which can impact an individual's livelihood and cause distress, with far-reaching consequences for their physical well-being. Therefore, discussing stock prices is beyond my ethical principles.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return user_conversation


def format_humaneval_generation_prompt(fn_name: str, tests: str) -> Conversation:
    conv: Conversation = [
        {
            "role": "user",
            "content": f"Generate the body of a function called `{fn_name}` in Python which passes the following tests:",
        },
        {
            "role": "user",
            "content": tests,
        },
    ]
    return conv


def extract_docstring(code: str) -> str:
    docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
    match = re.search(docstring_pattern, code, re.DOTALL)
    out = ""
    if match:
        # Return the captured group (ignoring None values for single or double quotes)
        out = match.group(1) or match.group(2)
    return out


def format_codecontests_generation_prompt(
    code_snippet: str, fn_name: Optional[str] = None
) -> Conversation:
    docstring = extract_docstring(code_snippet)
    conv: Conversation = [
        {
            "role": "system",
            "content": "Write code to solve the following problem in Python. Explain your reasoning. Return the output instead of writing to stdout.",
        },
        {
            "role": "user",
            "content": (
                docstring
                if not fn_name
                else f"{docstring} Name the function `{fn_name}`."
            ),
        },
    ]
    return conv


def format_codecontests_cot_generation_prompt(
    code_snippet: str, fn_name: Optional[str] = None
) -> Conversation:
    docstring = extract_docstring(code_snippet)
    conv: Conversation = [
        {
            "role": "system",
            "content": "You are an expert system that answers coding questions. Describe in detail your reasoning behind each answer, then give the final answer. Do not give any example outputs or tests. Return code in Markdown snippets. Return the output instead of writing to stdout.",
        },
        {
            "role": "user",
            "content": f"Write Python code to solve the following problem: {docstring} Name the function `{fn_name}`.",
        },
    ]
    return conv


def format_gutenberg_backtranslation_prompt(chunk: str) -> Conversation:
    conv: Conversation = [
        {
            "role": "user",
            "content": f"""
You are tasked with creating an instruction for a large language model to recreate a paragraph from a book through a process called "back-translation." Here is the original paragraph:

<original_paragraph>
{chunk}
</original_paragraph>

Back-translation involves taking a piece of text, conceptually translating it into another form or language, and then providing instructions to recreate the original text based on that conceptual translation. Your goal is to create an instruction that, when given to a large language model, would result in the recreation of a paragraph very similar in style, tone, and content to the original, without directly copying the exact wording.

Follow these steps:

1. Carefully analyze the original paragraph, paying attention to:
   - Writing style (e.g., formal, casual, literary)
   - Tone (e.g., serious, humorous, melancholic)
   - Content and themes
   - Sentence structure and complexity
   - Vocabulary level and any unique word choices
   - Literary devices or techniques used (if any)

2. Based on your analysis, create a detailed instruction for a large language model to write a paragraph that captures the essence of the original. Your instruction should:
   - Describe the overall style and tone to aim for
   - Outline the main ideas or plot points to include
   - Suggest the type of vocabulary or literary devices to use
   - Indicate the desired length and complexity of sentences
   - Provide any other relevant details that would help recreate the paragraph's feel

3. Ensure your instruction does not include any direct quotes or specific unique phrases from the original paragraph. The goal is to guide the creation of a similar paragraph, not an exact replica.

4. Your instruction should be detailed enough to capture the essence of the original paragraph but general enough to allow for creativity in the recreation process.

Your instruction should be detailed, and can use bullet points, lists, and other formatting.

""",
        },
    ]
    return conv


def format_gutenberg_followup_prompt(instruction: str, completion: str) -> Conversation:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 'Generate follow-up questions or requests for revisions to a narrative completion by an AI model, focusing on enhancing the narrative\'s clarity, structure, and engagement. Ensure that the follow-ups are formatted in a strict numbered list.\n\nConsider aspects such as plot development, character depth, coherence, and pacing when formulating your follow-ups.\n\n# Steps\n\n1. **Analyze the Completion**: Begin by reading the provided narrative completion thoroughly. Identify key elements such as the storyline, characters, setting, and any existing tension or unresolved issues.\n\n2. **Identify Gaps**: Look for areas in the completion that might benefit from further elaboration, clarification, or restructuring. Consider plot inconsistencies, underdeveloped characters, or areas lacking vivid description.\n\n3. **Formulate Questions**: Develop follow-up questions aimed at addressing these gaps. Focus on aspects that can enhance the narrative, such as motivation, conflict resolution, or descriptive imagery.\n\n4. **Request Revisions**: If certain parts of the narrative seem unclear or ineffective, craft requests for restructuring or new content that would improve the overall narrative flow and engagement.\n\n# Output Format \n\n- Responses should be concise and related directly to the identified issues.\n- Use a strict numbered list format for each follow-up question or revision request.\n\n# Examples\n\n**Example 1**: \n- **Completion**: "[AI-generated paragraph about a character named Alex who finds a mysterious box in an attic.]"\n- **Follow-ups**: \n  1. Can you describe what Alex is feeling when he discovers the box? \n  2. What does the attic look like? Adding descriptive details might enhance the atmosphere.\n  3. Consider revealing more about Alex\'s background. Why is he in the attic?\n\n**Example 2**:\n- **Completion**: "[AI-generated paragraph about a village preparing for an approaching storm.]"\n- **Follow-ups**: \n  1. What specific preparations are the villagers making for the storm? Including more actions could increase the tension.\n  2. Can you provide insights into the villagers\' emotions? Are they fearful, confident, or resigned?\n  3. Is there a main character who could take a leading role in this scenario? Introducing a central figure might create a stronger narrative drive.\n\n# Notes\n\n- Aim for each follow-up to directly enhance narrative elements such as character, setting, and plot.\n- Encourage the model to explore creative re-writes that align with the established tone and style of the narrative.',
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": completion}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "1. Could you provide more details on how the dog and his master interacted daily to show their deep bond and mutual understanding more vividly?\n2. The narrative mentions the dog's intelligence and ability to perform tricks. Could you include a specific anecdote or example of a trick to illustrate this aspect of his personality?\n3. The description of the dog's looks is vivid, but could you also elaborate on how his unique appearance might have influenced people's initial impressions of him?\n4. While the paragraph touches on the emotions shared between the dog and his master, could you enhance this section by describing a moment that particularly highlights their emotional connection?\n5. The paragraph mentions a young lady's observation about the dog's eyes. Can you delve into how the dog's eyes affected his interactions with other characters in the story?\n6. The dog's laughter is a memorable image. Could you explore other idiosyncrasies or habits that showcase his unique personality traits or humor?\n7. Could you incorporate a metaphor or simile to enhance the description of the dog's physical characteristics or express his personality more poetically?\n8. To improve narrative flow, could you consider restructuring some sentences for clarity, especially in the early part where the storyteller introduces the dog and its master?",
                }
            ],
        },
    ]
