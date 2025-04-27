import re
from synthetic_data.creative_writing_bench.prompts import (
    CRITERIA_STRICT,
    JUDGING_PROMPT,
)
from synthetic_data.creative_writing_bench.slop_score import (
    calculate_slop_index_new,
    load_slop_list_to_set,
)
from synthetic_data.tasks import RunMode


class CreativeWritingBench:
    def __init__(self, run_mode: RunMode) -> None:
        super().__init__()
        self.run_mode = run_mode
        template_path = (
            "/dataset_files/"
            if run_mode == "modal"
            else "../dataset_files/"
            if run_mode == "notebook"
            else "./dataset_files/"
        )
        # 1. Load Slop Lists
        self.slop_words_set = load_slop_list_to_set(f"{template_path}/slop_list.json")
        self.slop_bigrams_set = load_slop_list_to_set(
            f"{template_path}/slop_list_bigrams.json"
        )
        self.slop_trigrams_set = load_slop_list_to_set(
            f"{template_path}/slop_list_trigrams.json"
        )

    def format_prompt(self, writing_prompt: str, model_response: str) -> str:
        """
        Format the judge prompt with the creative writing criteria and negative criteria.
        """
        prompt = JUDGING_PROMPT.format(
            test_model_response=model_response,
            writing_prompt=writing_prompt,
            creative_writing_criteria=CRITERIA_STRICT,
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

    def calculate_slop_index(self, text: str) -> float:
        return calculate_slop_index_new(
            text,
            self.slop_words_set,
            self.slop_bigrams_set,
            self.slop_trigrams_set,
            debug=False,
        )
