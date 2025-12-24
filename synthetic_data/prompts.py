import re
from typing import Optional

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
    return user_conversation  # ty:ignore[invalid-return-type]


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

    user_message = {
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
    ]  # ty:ignore[invalid-assignment]
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


def format_writing_backtranslation_prompt(original_paragraph: str) -> Conversation:
    conv: Conversation = [
        {
            "role": "user",
            "content": f"""
You are an expert in literary analysis and content creation. Your task is to analyze a given paragraph and then create detailed instructions for recreating a similar paragraph that captures its essence without directly copying it. This process is called "back-translation."

Here is the original paragraph you need to analyze:

{original_paragraph}

Please follow these steps to complete your task:

1. Analyze the original paragraph:
   Wrap your analysis in <analysis> tags as you examine the paragraph. Include the following elements:
   
   a) Brief summary (2-3 sentences)
   b) Writing style (e.g., formal, casual, literary)
   c) Tone (e.g., serious, humorous, melancholic)
   d) Content and themes
   e) Sentence structure and complexity
   f) Vocabulary level and any unique word choices
   g) Literary devices or techniques used (if any)

   For each element, quote specific sentences or phrases from the text to support your analysis. Be thorough in your examination.

2. Plan detailed instructions:
   Based on your analysis, use <instruction_planning> tags to craft an instruction for recreating a paragraph that captures the essence of the original. Your planning should include:
   
   a) Style and Tone:
   Describe the overall writing style and tone to aim for. Be specific about the mood and atmosphere the writer should create.

   b) Content and Themes:
   Outline the main ideas, plot points, or narrative elements to include. Focus on the actions and events occurring in the paragraph, as well as any character development or thematic elements.

   c) Sensory Details and Imagery:
   Instruct on the use of sensory details and imagery to bring the scene to life. Specify which senses should be engaged and how.

   d) Character Interaction (if applicable):
   If the original paragraph includes character interactions, provide guidance on how to recreate similar dynamics.

Present your final instruction in the following format:

<instruction>
<style_and_tone>
[Your detailed description of the style and tone]
</style_and_tone>

<content_and_themes>
[Your comprehensive outline of main ideas, plot points, or narrative elements]
</content_and_themes>

<structure>
[Your guidance on sentence structure, paragraph organization, and overall flow]
</structure>

<character_interaction>
[Your guidance on character dynamics, if applicable]
</character_interaction>

<setting_description>
[Your advice on describing the setting, if important]
</setting_description>
</instruction>

Remember to use your analysis to inform every aspect of your instruction, ensuring that the essence of the original paragraph is captured in your guidance.
""",
        },
    ]
    return conv


def format_classify_fiction_prompt(paragraph: str) -> Conversation:
    return [
        {
            "role": "system",
            "content": """
Classify a given piece of text to determine whether it is a passage of narrative fiction that includes elements of dialogue and action, or if it is not.

Consider the following when making your classification:

- **Narrative Fiction**: Look for elements of storytelling, such as a plot or characters in action.
- **Dialogue**: Identify the presence of conversations or spoken exchanges between characters.
- **Action**: Detect descriptive sequences depicting physical actions or events.

# Steps

1. **Read the Text**: Carefully read through the provided text to identify key elements.
2. **Identify Dialogue**: Look for quotation marks or other indicators of characters speaking.
3. **Identify Action**: Seek out descriptions of characters performing actions or events unfolding.
4. **Conclusion**: Decide if both dialogue and action are present indicative of narrative fiction.

# Output Format

- Output should be a single sentence classification: "Narrative Fiction" or "Not Narrative Fiction."

# Examples

### Example 1
**Input**:  
"John looked at Sarah and said, 'I can't believe this is happening.' He then rushed towards the door, pulling it open with force."

**Output**:  
"Narrative Fiction"

### Example 2
**Input**:  
"The study of quantum mechanics requires an understanding of various complex principles and mathematical concepts."

**Output**:  
"Not Narrative Fiction"

# Notes

- Be aware of texts that might not follow traditional narrative forms yet qualify due to presence of characters, dialogue, and action.
- Maintain objectivity, focusing only on the presence of required elements without inferring absent details.
""",
        },
        {
            "role": "user",
            "content": paragraph,
        },
    ]


def tags_to_instruction(tags: dict) -> str:
    tags_str = "\n".join([f"- {v}" for v in tags.values()])
    return f"""
You are an expert writer. Your task is to take the following tags and create a detailed piece of writing.
{tags_str}
    """
