import ast
import inspect
from typing import List, Optional
from pydantic.dataclasses import dataclass
import random


def ConvertUnits(amount, from_unit, to_unit):
    """
    Convert a quantity from one unit to another. Returns the converted weight. Available formats are pounds, kilograms, ounces, grams, meters, feet, inches, centimeters, and kilometers.
    """

    # Dictionary containing conversion factors for various units
    conversion_factors = {
        "pounds": {"kilograms": 0.453592, "ounces": 16},
        "kilograms": {"pounds": 2.20462},
        "ounces": {"pounds": 0.0625},
        "grams": {"kilograms": 0.001},
        "meters": {
            "feet": 3.28084,
            "inches": 39.3701,
            "centimeters": 100,
            "kilometers": 0.001,
        },
        "feet": {"meters": 0.3048, "inches": 12},
        "inches": {"meters": 0.0254, "feet": 0.0833333},
        "centimeters": {"meters": 0.01},
        "kilometers": {"meters": 1000},
    }

    # Check if from_unit and to_unit are compatible units
    if (
        from_unit not in conversion_factors
        or to_unit not in conversion_factors[from_unit]
    ):
        return "Conversion from {} to {} is not supported.".format(from_unit, to_unit)

    # Perform conversion
    converted_amount = amount * conversion_factors[from_unit][to_unit]
    return converted_amount


def Calculator(expression):
    """
    Evaluate a mathematical expression. Returns the result of the expression.
    """

    try:
        # Evaluating the expression using the eval() function
        result = eval(expression)
        return result
    except Exception as e:
        # Handling any errors that may occur during evaluation
        return "Error: {}".format(str(e))


TOOL_FUNCTIONS = {
    "ConvertUnits": ConvertUnits,
    "Calculator": Calculator,
}

TOOL_DESCRIPTIONS = {
    "ConvertUnits": "Convert a quantity from one unit to another. Returns the converted weight. Available formats are pounds, kilograms, ounces, grams, meters, feet, inches, centimeters, and kilometers.",
    "Calculator": "Evaluate a mathematical expression. Returns the result of the expression.",
}


DROPOUT_TYPES = ["tool_description", "tool_parameter", "available_tools", "reorder_params"]


@dataclass
class FunctionCall:
    fn_name: str
    parameters: List[str | int | float | bool]


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            arguments = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    arguments.append(arg.value)

            self.function_calls.append((node.func.id, arguments))


def get_fn_call_metadata(text: str) -> FunctionCall:
    parsed = ast.parse(text)
    visitor = FunctionCallVisitor()
    visitor.visit(parsed)
    call = FunctionCall(visitor.function_calls[0][0], visitor.function_calls[0][1])
    return call


def get_function_info(func):
    func_name = func.__name__
    sig = inspect.signature(func)
    args = [
        param.name
        for param in sig.parameters.values()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    return func_name, args


def get_tool_descriptions(dropout_type: Optional[str] = None):
    tool_description_lines = []
    if dropout_type == "available_tools":
        return ""
    for tool_name in TOOL_FUNCTIONS.keys():
        description = TOOL_DESCRIPTIONS[tool_name]
        if dropout_type == "tool_description":
            description = ""
        tool_name, tool_args = get_function_info(TOOL_FUNCTIONS[tool_name])
        if dropout_type == "tool_parameter":
            tool_args = []
        if dropout_type == "reorder_params":
            random.shuffle(tool_args)
        tool_description_lines += [
            f"{tool_name}({', '.join(tool_args)}): {description}"
        ]

    return "\n".join(tool_description_lines)
