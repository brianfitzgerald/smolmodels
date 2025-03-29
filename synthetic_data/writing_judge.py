from synthetic_data.generation import Conversation
import re


def parse_judge_scores_creative(judge_model_response: str) -> dict[str, float]:
    scores = {}

    # Parse scores using multiple regex patterns
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)"
    # Pattern 2: Metric: [Score]
    score_pattern2 = r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]"

    # Combine both patterns
    matches1 = re.findall(score_pattern1, judge_model_response)
    matches2 = re.findall(score_pattern2, judge_model_response)

    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            score = float(match[1])
            scores[metric_name] = score

    return scores


class CreativeWritingBench:
    def __init__(self) -> None:
        super().__init__()
        with open("dataset_files/negative_criteria.txt", "r", encoding="utf-8") as f:
            self.negative_criteria = [line.strip() for line in f if line.strip()]
        with open(
            "dataset_files/creative_writing_criteria.txt", "r", encoding="utf-8"
        ) as f:
            self.creative_writing_criteria = [
                line.strip() for line in f if line.strip()
            ]
        with open(
            "dataset_files/creative_writing_judging_prompt.txt", "r", encoding="utf-8"
        ) as f:
            self.judge_prompt_template = f.read()

    def format_prompt(self, model_text: str) -> str:
        """
        Format the judge prompt with the creative writing criteria and negative criteria.
        """
        return self.judge_prompt_template.format(
            creative_writing_criteria="\n".join(
                ["- " + c for c in self.creative_writing_criteria]
            ),
            negative_criteria=", ".join(self.negative_criteria),
            test_model_response=model_text,
        )

    def parse_judge_scores(self, judge_model_response: str) -> dict[str, float]:
        scores = parse_judge_scores_creative(judge_model_response)
        return scores
