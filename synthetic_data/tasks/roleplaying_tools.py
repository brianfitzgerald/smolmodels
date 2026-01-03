"""
Tool definitions and execution functions for the roleplaying game task.
Supports both Dungeon Master and Player roles with native API tool calling.
"""

import json
import random
import re

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from synthetic_data.utils import ToolParam, ToolResultBlock, ToolUseBlock

ToolOutput = dict[str, str | int | list[str] | bool]

# Dice roll pattern: matches "1d20", "2d6+3", "d20", "3d8-2", etc.
DICE_PATTERN = re.compile(r"^(\d*)d(\d+)([+-]\d+)?$", re.IGNORECASE)


ROLL_DICE_TOOL: ToolParam = {
    "name": "roll_dice",
    "description": "Roll dice using standard RPG notation. Use this for any random chance events, skill checks, combat, or when randomness is needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "notation": {
                "type": "string",
                "description": "Dice notation like '2d6+3' (roll 2 six-sided dice and add 3), '1d20' (roll one 20-sided die), '3d8-2' (roll 3 eight-sided dice and subtract 2). Format: [count]d[sides][+/-modifier]",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of what this roll is for (e.g., 'attack roll', 'perception check', 'damage')",
            },
        },
        "required": ["notation", "reason"],
    },
}

RANDOM_CHOICE_TOOL: ToolParam = {
    "name": "random_choice",
    "description": "Randomly select one option from a list. Use this when the outcome should be randomly determined from multiple possibilities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of options to randomly choose from",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of what this choice determines",
            },
        },
        "required": ["options", "reason"],
    },
}

SPEAK_TOOL: ToolParam = {
    "name": "speak",
    "description": "Have a character speak dialogue. Use this for NPC dialogue (as DM) or player character dialogue (as player).",
    "input_schema": {
        "type": "object",
        "properties": {
            "character": {
                "type": "string",
                "description": "Name of the character speaking",
            },
            "message": {"type": "string", "description": "What the character says"},
            "tone": {
                "type": "string",
                "description": "Optional tone or emotion (e.g., 'whispered', 'shouted', 'nervously')",
            },
        },
        "required": ["character", "message"],
    },
}

ACTION_TOOL: ToolParam = {
    "name": "action",
    "description": "Perform a general action in the game world. Use this for player actions that interact with the environment or other characters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of the action being taken. Keep the description concise and describe in first person, such as 'Walk forward' or 'Look around'",
            },
        },
        "required": ["description"],
    },
}

# Tool collections by role
DM_TOOLS: list[ToolParam] = [ROLL_DICE_TOOL, RANDOM_CHOICE_TOOL, SPEAK_TOOL]
PLAYER_TOOLS: list[ToolParam] = [SPEAK_TOOL, ACTION_TOOL]

# All tools combined
ALL_TOOLS: list[ToolParam] = [
    ROLL_DICE_TOOL,
    RANDOM_CHOICE_TOOL,
    SPEAK_TOOL,
    ACTION_TOOL,
]


def parse_dice_notation(notation: str) -> tuple[int, int, int] | None:
    """Parse dice notation into (num_dice, num_sides, modifier).

    Returns None if the notation is invalid.
    """
    match = DICE_PATTERN.match(notation.strip())
    if not match:
        return None

    num_dice = int(match.group(1)) if match.group(1) else 1
    num_sides = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0

    return num_dice, num_sides, modifier


def execute_roll_dice(notation: str, reason: str) -> dict:
    """Execute a dice roll and return the result."""
    parsed = parse_dice_notation(notation)
    if parsed is None:
        return {
            "error": f"Invalid dice notation: {notation}",
            "success": False,
        }

    num_dice, num_sides, modifier = parsed
    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    total = sum(rolls) + modifier

    return {
        "notation": notation,
        "reason": reason,
        "rolls": rolls,
        "modifier": modifier,
        "total": total,
    }


def execute_random_choice(options: list[str], reason: str) -> ToolOutput:
    """Execute a random choice from options."""
    if not options:
        return {
            "error": "No options provided",
            "success": False,
        }

    chosen = random.choice(options)
    return {
        "reason": reason,
        "options": options,
        "chosen": chosen,
        "index": options.index(chosen),
    }


def execute_speak(character: str, message: str, tone: str | None = None) -> ToolOutput:
    """Execute a speak action."""
    result: ToolOutput = {
        "character": character,
        "message": message,
    }
    if tone:
        result["tone"] = tone
    return result


def execute_action(description: str, target: str | None = None) -> ToolOutput:
    """Execute a player action."""
    result: ToolOutput = {
        "description": description,
        "executed": True,
    }
    if target:
        result["target"] = target
    return result


TOOL_EXECUTORS = {
    "roll_dice": lambda args: execute_roll_dice(args["notation"], args["reason"]),
    "random_choice": lambda args: execute_random_choice(
        args["options"], args["reason"]
    ),
    "speak": lambda args: execute_speak(
        args["character"], args["message"], args.get("tone")
    ),
    "action": lambda args: execute_action(args["description"], args.get("target")),
}


def execute_tool_call(tool_call: ChatCompletionMessageToolCall) -> ToolResult:
    """Execute a tool call and return the result."""
    tool_name = tool_call.function.name
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return {
            "error": f"Unknown tool: {tool_name}",
            "success": False,
        }

    # Parse the arguments from JSON string
    arguments = json.loads(tool_call.function.arguments)
    result = executor(arguments)
    result["tool_call_id"] = tool_call.id
    return result


def execute_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
) -> list[ToolOutput]:
    """Execute multiple tool calls and return results."""
    return [execute_tool_call(tc) for tc in tool_calls]


def execute_tool_use_block(tool_use: ToolUseBlock) -> ToolResultBlock:
    """
    Execute a tool_use block and return the result as a JSON string.

    This function matches the signature expected by GenerationArgs.tool_use_executor:
    Callable[[ToolUseBlock], str]
    """
    tool_name = tool_use.get("name", "")
    tool_input = tool_use.get("input", {})

    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return ToolResultBlock(
            type="tool_result",
            tool_use_id=tool_use.get("id", ""),
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        # tool_input might be a dict or might need parsing
        if isinstance(tool_input, str):
            args = json.loads(tool_input)
        else:
            args = tool_input

        result = executor(args)
        return ToolResultBlock(
            type="tool_result",
            tool_use_id=tool_use.get("id", ""),
            content=json.dumps(result.content),
            is_error=result.success,
        )
    except Exception as e:
        return ToolResultBlock(
            type="tool_result",
            tool_use_id=tool_use.get("id", ""),
            content=str(e),
            is_error=True,
        )
