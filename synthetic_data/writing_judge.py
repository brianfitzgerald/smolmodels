from synthetic_data.generation import Conversation

JUDGING_CRITERIA = [
    {
        "prefix_text": "Now, rate the supplied model output on the following criteria:",
        "criteria": [
            "Overall Impression",
            "Overall Reader Engagement",
            "Sentences Flow Naturally",
            "Well-earned Lightness or Darkness",
        ],
    },
    {
        "prefix_text": "Now, rate the supplied model output on the following criteria (lower = better):",
        "criteria": [
            "Unearned Transformations",
            "Incongruent Ending Positivity",
            "Overwrought",
            "Purple Prose",
            "Amateurish",
            "Unsurprising or Uncreative",
            "Tell-Don't-Show",
            "Weak Dialogue",
            "Meandering",
        ],
    },
    {
        "prefix_text": "Now, rate the supplied model output on the following criteria:",
        "criteria": [
            "Coherent",
            "Emotionally Complex",
            "Emotionally Engaging",
            "Elegant Prose",
            "Imagery and Descriptive Quality",
        ],
    },
    {
        "prefix_text": "Now, rate the supplied model output on the following criteria:",
        "criteria": [
            "Consistent Voice/Tone of Writing",
            "Nuanced Characters",
            "Believable Character Actions",
            "Adherence to Instructions",
        ],
    },
]


def create_judging_prompt(
    criteria_set, writing_prompt, reference_output, test_model_response
):
    criteria = [x for x in criteria_set["criteria"]]
    prefix_text = criteria_set["prefix_text"]
    criteria_str = "\n".join(criteria)

    negative_criteria_section = """
- For these criteria, lower is better:
Unearned Transformations
Incongruent Ending Positivity
Overwrought
Purple Prose
Amateurish
Unsurprising or Uncreative
Tell-Don't-Show
Weak Dialogue
Meandering
"""

    analysis_section_1 = """
- You are to write a comprehensive analysis of the piece, then give your scores.
"""
    analysis_section_2 = """
[Analysis]

Write your detailed analysis.
"""
    ref_str = ""
    relative_section_1 = f"""You are an expert in assessing creative writing. Your task is to score the quality of the test model's response above, by several metrics, on a 0-10 scale.

Scoring notes:
{ref_str}
- Scores of 0 or 10 should not be considered highly unlikely just because they are the max/min. Use the full scoring range as appropriate.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment."""
    relative_section_2 = "[Score 0-10]"

    reference_section_1 = ""
    # Construct judging prompt
    judging_prompt = f"""
You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-10 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]
{reference_section_1}
[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

{relative_section_1}

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Some models produce overly long outputs. You should neither penalise nor favour this if it happens; simply assess the writing on its merit. You should however penalise overly short pieces.

- The test model's output can suddenly truncate because of token length constraints. If you notice that this has occurred, don't penalise it.

- Do not use markdown in your response. Use the designated output format exactly.

- Some models have a positivity bias that produces worse writing. You'll know it when you see it (particularly with unearned positive resolutions).
{negative_criteria_section}
- You are a critic, so be honest, objective, critical and discriminative. No need to be charitable; say what you genuinely think.
{analysis_section_1}
- Output format is:
{analysis_section_2}
[Scores]

Metric 1 name: {relative_section_2}

Metric 2 name: ...

---

{prefix_text}

{criteria_str}
	"""
    return judging_prompt


def format_gutenberg_judge_prompt(instruction: str, completion: str) -> Conversation:
    return [
        {
            "role": "system",
            "content": create_judging_prompt(
                JUDGING_CRITERIA[0], instruction, "", completion
            ),
        },
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": completion},
    ]
