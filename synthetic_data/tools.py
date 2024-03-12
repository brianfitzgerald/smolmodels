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