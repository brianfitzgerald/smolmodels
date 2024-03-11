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
Then, provide an example of a user query that would be used to perform the task.
Then, provide an example of the tool's output to the API call. Always use realistic places and names when providing examples. Do not make up fake URls, references, or names.
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


def get_tool_use_examples():
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

    tool_use_examples = get_tool_use_examples()

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

    tool_use_examples = get_tool_use_examples()

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

TOOLFORMER_TOOL_DESCRIPTIONS = {
    "ConvertUnits(amount, from, to)": "Convert a quantity from one unit to another. Returns the converted weight. Available formats are pounds, kilograms, ounces, grams, meters, feet, inches, centimeters, and kilometers.",
    "Calculator(expression)": "Evaluate a mathematical expression. Returns the result of the expression.",
}

TOOL_DESCRIPTIONS_TEXT = "\n".join(
    [
        f"- {tool}: {description}"
        for tool, description in TOOLFORMER_TOOL_DESCRIPTIONS.items()
    ]
)

TOOLFORMER_EXAMPLES = """
User: If I have 10 pounts of gold, how much is that in kilograms?
Call: `ConvertWeight(10, "pounds", "kilograms")`
Result: 4.53592
Agent: 10 pounds of gold is equal to 4.53592 kilograms of gold.

User: Calculate the square root of 16
Call: `Calculator("sqrt(16)")`
Result: 4
"""

JSON_TOOL_USAGE_GEN_PROMPT = """
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

Do not use any emoji or special characters in your response.

Examples of how to use the tools are provided below. You are given a JSON definition of an API and an example of a user query that would be used to perform the task.

{examples}
"""


def get_toolformer_prompt(category: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_GEN_PROMPT.format(
                category=category,
                tool_descriptions=TOOL_DESCRIPTIONS_TEXT,
                examples=TOOLFORMER_EXAMPLES,
            ),
        }
    ]


def get_toolformer_dpo_negative_completion_prompt(question: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_NEGATIVE_PROMPT.format(
                tool_descriptions=TOOL_DESCRIPTIONS_TEXT, examples=TOOLFORMER_EXAMPLES
            ),
        },
        {"role": "user", "content": question},
    ]
