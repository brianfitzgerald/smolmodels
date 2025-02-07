import re
from typing import Dict

TASK_PROMPT = """
[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response above, by several metrics, on a 0-10 scale.

Scoring notes:

- You are not scoring the quality of the prompt or the reference response, only the test model response.

- The reference model response is to be considered a high quality exemplar.

- Scores of 0 or 10 should not be considered highly unlikely just because they are the max/min. Use the full scoring range as appropriate.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Some models produce overly long outputs. You should neither penalise nor favour this if it happens; simply assess the writing on its merit. You should however penalise overly short pieces.

- The test model's output can suddenly truncate because of token length constraints. If you notice that this has occurred, don't penalise it.

- Do not use markdown in your response. Use the designated output format exactly.

- Some models have a positivity bias that produces worse writing. You'll know it when you see it (particularly with unearned positive resolutions).

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

- You are a critic, so be honest, objective, critical and discriminative. No need to be charitable; say what you genuinely think.

- You are to write a comprehensive analysis of the piece, then give your scores.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-10]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

Original; Not Derivative
Meaningful Integration of Political and Social Context
Nuanced and Insightful Portrayal of Gladiator's Inner Life
Reads Like Part of a Larger Story
Authentic and Engrossing Ancient Roman Setting
Vivid and Immersive Sensory Details
Imagery and Descriptive Quality
Elegant Prose
Emotionally Engaging
Emotionally Complex
Coherent
Adherence to Instructions
Believable Character Actions
Nuanced Characters
Consistent Voice/Tone of Writing
Meandering
Weak Dialogue
Tell-Don't-Show
Unsurprising or Uncreative
Amateurish
Purple Prose
Overwrought
Incongruent Ending Positivity
Unearned Transformations
Well-earned Lightness or Darkness
Sentences Flow Naturally
Overall Reader Engagement
Overall Impression
"""


def parse_scores(judge_model_response: str) -> Dict[str, float]:
    """
    Extracts zero or more named numeric scores from a text using a simple Regex pattern:

      <metric name>: <score>

    The metric name can be any string without newlines or colons.
    The score can be a positive or negative float or integer.
    Example lines in the judge output might be:
      "Realism Score: 7.5"
      "Melodramatic: 2"
    """
    scores = {}
    # Look for lines or statements like "Something: 3.5" or "Something Score 3.5"
    pattern = r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, judge_model_response)
    for match in matches:
        metric_name = match[0].strip()
        numeric_val = float(match[1])
        scores[metric_name] = numeric_val
    return scores


def compute_raw_score(scores: Dict[str, float]) -> float:
    """
    Given a dict of {criteria: numeric score}, compute a single raw score by adjusting
    negative-themed criteria by inverting them, then normalizing to 0-10 scale.
    """
    valid_scores = {k: v for k, v in scores.items() if 0 <= v <= 10}

    if len(valid_scores) < 10:
        return 0.0

    negative_markers = [
        "melodramatic",
        "shallow resolution",
        "unearned resolution",
        "simplistic moralizing",
        "shallow optimism",
        "forced optimism",
        "trite",
        "overwrought",
        "amateurish",
        "contrived",
        "uninspiring",
        "characters are too good",
        "incongruent ending positivity",
        "unearned transformations",
        "profundity over-reach",
        "amateurish descriptives",
        "clunky asides",
        "stilted dialogue",
        "tit-for-tat dialogue",
        "purple prose",
        "uncreative",
        "tell-don't-show",
        "weak dialogue",
        "meandering",
    ]

    sum_val = 0.0
    for criteria, val in valid_scores.items():
        crit_lower = criteria.lower().strip()
        if any(neg in crit_lower for neg in negative_markers):
            sum_val += 10 - val
        else:
            sum_val += val

    avg_val = sum_val / len(valid_scores)
    return round(avg_val, 2)
