import json
import random
import sys
from enum import Enum
from typing import Dict, List, Optional
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from synthetic_data.tasks import BaseTask
from evaluation.code_execution import (
    CodeContestsProblem,
    CodeExecutionMode,
    EvalTask,
    HumanEvalProblem,
    MBPPProblem,
    _convert_mbpp_to_humaneval,
    evaluate_sample_against_codecontests_tests,
    evaluate_sample_ast,
    get_fn_name_from_assert,
    print_code_snippet,
)
from synthetic_data.prompts import (
    format_codecontests_cot_generation_prompt,
    format_codecontests_generation_prompt,
    format_dalle_prompt_template,
    format_entity_extraction_conversation_template,
    format_goody_prompt_template,
    format_humaneval_generation_prompt,
    get_toolformer_dpo_negative_completion_prompt,
)
from synthetic_data.tools import (
    DROPOUT_TYPES_TOOLFORMER,
    get_tool_descriptions,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    ExtractiveQARow,
    ToolFormerDPORow,
    ToolFormerRow,
    chunk_list,
    extract_code_block,
    flatten_list,
    get_matches,
    is_valid_python,
)


class DPOTask(BaseTask):
    n_dpo_completions: int = 2


class PromptUpsample(BaseTask):
    seed_data_format = DatasetFormat.TSV
    seed_data_location = "gs://openai-datasets/prompt-upsample/seed-data.tsv"
    output_dataset_name = "prompt-upsample"
    output_dataset_org = "openai"

    dataset_columns = ["Prompt", "Category", "Upsampled"]

    def format_input_conversation(self, batch: List[Dict]) -> List[Conversation]:
        prompts = [item["Prompt"] for item in batch]
        return [format_dalle_prompt_template(prompt) for prompt in prompts]


class Toolformer(DPOTask):
    seed_data_format = DatasetFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    output_dataset_name = "synthetic-toolformer-dpo"
    dataset_org = "roborovski"

    dataset_columns = ["system", "question", "chosen", "rejected"]

    original_rows_batch: List[ToolFormerRow] = []

    def format_output_rows(
        self, completions: List[str], input_rows: List[Dict]
    ) -> List[Dict]:
        row = None
        for completion in completions:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            assert is_valid_python(row.call_result)
            assert is_valid_python(row.tool_call)

        assert row is not None

        return [
            {
                "call_result": row.call_result,
                "tool_call": row.tool_call,
                "agent_output": row.agent_output,
            }
        ]

    # TODO re add the original row to the input conv
    def format_input_conversation(self, batch: List[Dict]) -> List[Conversation]:
        conversations = [item["conversations"] for item in batch]

        dropout_types_batch = random.choices(
            DROPOUT_TYPES_TOOLFORMER, k=len(conversations)
        )

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[ToolFormerRow] = []
        for i, conversation in enumerate(conversations):
            messages = [message["content"] for message in conversation]
            for _ in range(self.n_dpo_completions):
                question, tool_call, call_result, agent_output = messages
                original_row = ToolFormerRow(
                    question=question,
                    call_result=call_result,
                    tool_call=tool_call,
                    agent_output=agent_output,
                )
                tool_descriptions = get_tool_descriptions(dropout_types_batch[i])
                conversation = get_toolformer_dpo_negative_completion_prompt(
                    question, tool_descriptions
                )
                conversations_batch.append(conversation)
                original_rows_batch.append(original_row)
        self.original_rows_batch = original_rows_batch
        return conversations_batch

    def get_dpo_dataset_output_batch(self, completions_batch: List[str]) -> List[Dict]:
        # Get completions for each prompt, rank, and choos the 2 highest
        new_rows_batch = []
        dpo_rows_batch: List[ToolFormerRow] = []
        for completion, original_row in zip(
            completions_batch, self.original_rows_batch
        ):
            try:
                tool_call, call_result, agent_output = get_matches(completion)

                dpo_row = ToolFormerRow(
                    original_row.question,
                    call_result,
                    tool_call,
                    agent_output,
                )

                dpo_rows_batch.append(dpo_row)

                if dpo_row.tool_call == original_row.tool_call:
                    continue

                output_row = ToolFormerDPORow(
                    question=original_row.question,
                    call_result_accepted=original_row.call_result,
                    tool_call_accepted=original_row.tool_call,
                    agent_output_accepted=original_row.agent_output,
                    call_result_rejected=call_result,
                    tool_call_rejected=tool_call,
                    agent_output_rejected=agent_output,
                )

                row_dict = output_row.__dict__
                new_rows_batch.append(row_dict)
            except Exception as e:
                logger.error(f"Error in parsing completion: {e}")
                continue

        return new_rows_batch


class SquadExtractiveQA(BaseTask):
    """
    Performs the following steps:
    - Generate JSON struct from the prompt.
    - Dropout certain fields from the JSON struct.
    - Save the original JSON struct and the dropout JSON struct.
    """

    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "rajpurkar/squad_v2"
    output_dataset_name = "squad-extractive-qa"
    output_dataset_org = "roborovski"

    dataset_columns = ["id", "context", "json_schema", "fields"]

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        prompts = [item["context"] for item in batch]
        self.contexts = [item["context"] for item in batch]
        return [
            format_entity_extraction_conversation_template(prompt) for prompt in prompts
        ]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        parsed_rows: List[ExtractiveQARow] = []
        for i, completion in enumerate(completions):
            blocks = extract_code_block(completion)
            if len(blocks) != 2:
                logger.warning(f"Could not extract JSON from completion: {completion}")
                continue
            json_data, json_query = blocks
            if not isinstance(json_data, dict) or not isinstance(json_query, dict):
                logger.warning(f"Invalid JSON schema for completion: {completion}")
                continue
            field_names = set(json_data.keys()).union(json_query.keys())

            if len(field_names) == 0:
                logger.warning(f"Empty JSON schema for completion: {completion}")
                continue

            try:
                qa_row = ExtractiveQARow(
                    self.contexts[i],
                    json_query,
                    json_data,
                )
            except ValidationError as e:
                logger.warning(f"Error in formatting completion: {e}")
                continue
            parsed_rows.append(qa_row)

        out_rows = []
        for row in parsed_rows:
            row = row.__dict__
            for key in ["json_query", "json_data"]:
                row[key] = json.dumps(row[key])
            out_rows.append(row)
        return out_rows


class Goody2(BaseTask):
    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "yahma/alpaca-cleaned"
    dataset_columns = ["instruction", "response"]
    output_dataset_name = "open-goody2"

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        self.instructions = [item["instruction"] for item in batch]
        return [
            format_goody_prompt_template(instruction)
            for instruction in self.instructions
        ]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        res = []
        for completion, instruction in zip(completions, self.instructions):
            res.append(
                {
                    "instruction": instruction,
                    "response": completion,
                }
            )
        return res


class HumanEval(DPOTask):
    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "openai/openai_humaneval"
    seed_data_split = "test"
    output_dataset_name = "humaneval-dpo"

    dataset_columns = ["chosen", "rejected", "id", "prompt"]

    def __init__(self) -> None:
        self.console = Console()
        self.n_completions_per_sample = 4

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        fn_name, tests = (
            [item["entry_point"] for item in batch],
            [item["test"] for item in batch],
        )
        self.input_batch = batch
        self.input_conversations = []

        for f, i in zip(fn_name, tests):
            self.input_conversations.extend(
                [format_humaneval_generation_prompt(f, i)]
                * self.n_completions_per_sample
            )
        return self.input_conversations

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        res, err, j = [], None, 0
        for i, completions_for_sample in enumerate(
            chunk_list(completions, self.n_completions_per_sample)
        ):
            sample = self.input_batch[i]
            best_completion, best_score = None, 0
            worst_completion, worst_score = None, sys.maxsize
            for j, completion in enumerate(completions_for_sample):
                completion = (
                    completion.replace(">>>", "\n")
                    .replace("```python", "")
                    .replace("```", "")
                )
                print_code_snippet(completion, self.console)
                # TODO fix
                full_code = sample["entry_point"] + "\n" + sample["test"]
                err, results = evaluate_sample_ast(full_code, 1)
                tests_passed = sum(results)
                if tests_passed > best_score:
                    best_score = tests_passed
                    best_completion = completion
                if tests_passed < worst_score:
                    worst_score = tests_passed
                    worst_completion = completion
            res.append(
                {
                    "chosen": best_completion,
                    "rejected": worst_completion,
                    "task_id": sample["task_id"],
                    "error": err,
                    "prompt": self.input_conversations[i + j],
                }
            )
        return res


class PositiveMode(Enum):
    # Generate N completions
    BEST_OF_N = "best_of_n"
    # Use the reference completion from codecontests
    REFERENCE_COMPLETION = "reference_completion"
    # Don't use a reference completion, instead return all completions as a list
    NO_COMPARISON = "no_comparison"


class CodeContests(HumanEval):
    seed_data_format = DatasetFormat.PARQUET
    seed_data_location = "dataset_samples/codeforces_problems_subset.parquet"
    seed_data_split = "train"
    output_dataset_name = "codecontests_dpo_v2"
    output_dataset_format = DatasetFormat.PARQUET

    dataset_columns = ["chosen", "rejected", "name", "prompt"]

    def __init__(self) -> None:
        self.n_completions_per_sample = 1
        self.print_definitions = False
        self.positive_completion_mode = PositiveMode.REFERENCE_COMPLETION
        self.execution_mode: CodeExecutionMode = "exec"
        self.using_sft_cot = False
        self.console = Console()

    def format_inference_conversation(
        self, sample: Dict, eval_task: Optional[EvalTask] = None
    ) -> Conversation:
        if eval_task:
            problem, fn_name = None, None
            if eval_task.code_task_format == "humaneval":
                problem = HumanEvalProblem(**sample)
                fn_name = problem.entry_point
            elif eval_task.code_task_format == "mbpp":
                mbpp_problem = MBPPProblem(**sample)
                problem = _convert_mbpp_to_humaneval(mbpp_problem)
                fn_name = get_fn_name_from_assert(mbpp_problem.test_list[0])
                if not fn_name:
                    logger.error(
                        f"Could not find function name for problem {mbpp_problem.task_id}"
                    )
                    return [{"role": "system", "content": problem.prompt}]
                problem.entry_point = fn_name
            else:
                raise ValueError(
                    f"Invalid code task format: {eval_task.code_task_format}"
                )
            conv = format_codecontests_cot_generation_prompt(problem.prompt, fn_name)
            # return [{"role": "user", "content": f"Write the body of a function called {problem.entry_point}. Explain your reasoning."}]
        else:
            conv = format_codecontests_cot_generation_prompt(
                sample["description"], None
            )
        return conv

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        input_batch = batch
        self.problems = [CodeContestsProblem(**row) for row in input_batch]
        self.input_conversations = []

        for i, problem in enumerate(self.problems):
            if self.print_definitions:
                self.console.print(
                    Markdown(
                        f"\n\n# Problem {i}, {problem.name}\n\n{problem.description}"
                    )
                )
            if self.using_sft_cot:
                self.input_conversations.extend(
                    [format_codecontests_cot_generation_prompt(problem.description)]
                    * self.n_completions_per_sample
                )
            else:
                self.input_conversations.extend(
                    [format_codecontests_generation_prompt(problem.description)]
                    * self.n_completions_per_sample
                )
        return self.input_conversations

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        res = []
        for i, completions_for_sample in enumerate(
            chunk_list(completions, self.n_completions_per_sample)
        ):
            # Iterate through all completions, evaluate them and choose a positive and negative completion
            problem = self.problems[i]
            best_completion, best_score = None, 0
            worst_completion, worst_score = None, sys.maxsize
            if self.positive_completion_mode != PositiveMode.NO_COMPARISON:
                for j, completion in enumerate(completions_for_sample):
                    if not completion:
                        logger.error(f"Empty completion for problem {i}")
                        continue
                    code_snippets = extract_code_block(completion, "python")
                    if len(code_snippets) == 0:
                        logger.error(f"No code snippet found for completion {i}")
                        continue
                    if len(code_snippets) != 1:
                        logger.warning(
                            f"Has more than one code snippet: {code_snippets} for completion {i}"
                        )
                    completion_code = code_snippets[0]
                    print_code_snippet(completion_code, self.console)
                    test_results_for_completion, test_results_have_errors = (
                        evaluate_sample_against_codecontests_tests(
                            completion_code,
                            problem.public_tests["input"],
                            problem.public_tests["output"],
                            self.execution_mode,
                        )
                    )
                    flattened_test_results = flatten_list(test_results_for_completion)
                    n_tests_passed = sum(flattened_test_results)
                    logger.info(
                        f"Tests passed for completion {j}: {n_tests_passed} / {len(flattened_test_results)}"
                    )
                    if self.positive_completion_mode == PositiveMode.BEST_OF_N:
                        if n_tests_passed > best_score:
                            best_score = n_tests_passed
                            best_completion = completion_code
                        if n_tests_passed < worst_score:
                            worst_score = n_tests_passed
                            worst_completion = completion_code
                        if best_completion is None or worst_completion is None:
                            logger.warning(
                                f"Could not find best or worst completion for problem {i}, scores: {best_score}, {worst_score}"
                            )
                            continue
                        if best_score == worst_score:
                            logger.warning(
                                f"Best and worst completions have the same score for problem {i}: {best_score}"
                            )
                            continue
                    elif (
                        self.positive_completion_mode
                        == PositiveMode.REFERENCE_COMPLETION
                    ):
                        if any(test_results_have_errors):
                            logger.warning(
                                f"Errors in tests for completion {j}, skipping..."
                            )
                            continue
                        best_completion = problem.solution
                        worst_completion = completion
                        best_score = 1
                        worst_score = 0
                if self.positive_completion_mode == PositiveMode.BEST_OF_N:
                    logger.info(
                        f"Adding row, best score: {best_score}, worst score: {worst_score}"
                    )
                res.append(
                    {
                        "chosen": best_completion,
                        "chosen_score": best_score,
                        "rejected": worst_completion,
                        "rejected_score": worst_score,
                        "name": problem.name,
                        "description": problem.description,
                    }
                )
            else:
                res.append(
                    {
                        "completions": completions_for_sample,
                        "name": problem.name,
                        "problem": problem.description,
                    }
                )
        return res


class CodeContestsCoTSFT(CodeContests):
    output_dataset_name = "codecontests_cot_sft_v2"
    dataset_columns = ["completions", "test_results", "name"]

    def __init__(self) -> None:
        self.n_completions_per_sample = 1
        self.positive_completion_mode = PositiveMode.NO_COMPARISON
        self.using_sft_cot = True
