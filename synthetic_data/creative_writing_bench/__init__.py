import re
from synthetic_data.creative_writing_bench.prompts import (
    JUDGING_CRITERIA,
    JUDGING_PROMPT,
)


class CreativeWritingBench:
    def __init__(self) -> None:
        super().__init__()

    def format_prompt(self, writing_prompt: str, model_response: str) -> str:
        """
        Format the judge prompt with the creative writing criteria and negative criteria.
        """
        prompt = JUDGING_PROMPT.format(
            test_model_response=model_response,
            writing_prompt=writing_prompt,
            creative_writing_criteria=JUDGING_CRITERIA,
        )
        return prompt

    def parse_judge_scores(self, judge_model_response: str) -> dict[str, float]:
        scores = {}

        score_pattern1 = r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)"
        score_pattern2 = r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]"

        matches1 = re.findall(score_pattern1, judge_model_response)
        matches2 = re.findall(score_pattern2, judge_model_response)

        for matches in [matches1, matches2]:
            for match in matches:
                metric_name = match[0].strip()
                score = float(match[1])
                scores[metric_name] = score

        return scores
