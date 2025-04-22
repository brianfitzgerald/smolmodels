import json
import os
from typing import Dict, List, Literal

import pandas as pd
from fire import Fire
from rich.console import Console
from rich.progress import Progress
import random
from datetime import datetime

from evaluation.code_execution import (
    EvalResult,
    EvalTask,
)
from synthetic_data.generation import (
    RemoteModel,
    get_generation_wrapper,
)
from synthetic_data.tasks.evals import EQBenchWriting
from synthetic_data.utils import ensure_directory


def _save_eval_results_to_csv(eval_results: List[EvalResult], out_dir: str):
    test_results_dicts = []
    for res in eval_results:
        test_results_dicts.append(
            {
                "prompt": res.prompt,
                "model_response": res.model_response,
                "scores": res.scores,
                "error": res.error,
            }
        )

    test_results_pd = pd.DataFrame(test_results_dicts)
    test_results_pd.to_csv(f"{out_dir}/test_results.csv", index=False)


def _save_eval_results_to_md(eval_results: List[EvalResult], out_dir: str):
    md_out_lines = []
    for i, res in enumerate(eval_results):
        md_out_lines.append(f"## Result {i}:")
        md_out_lines.append(f"**Prompt:** {res.prompt}")
        md_out_lines.append(f"**Model Response:** {res.model_response}")
        md_out_lines.append(f"**Scores:** {res.scores}")
        md_out_lines.append(f"**Error:** {res.error}")

    with open(f"{out_dir}/eval_results.md", "w") as f:
        f.write("\n".join(md_out_lines))


EvalTaskName = Literal["eq_bench_writing"]

EVAL_TASKS: Dict[EvalTaskName, type[EvalTask]] = {
    "eq_bench_writing": EQBenchWriting,
}


async def main(
    batch_size: int = 8,
    eval_task_name: EvalTaskName = "eq_bench_writing",
    gen_source: RemoteModel = "gemini-2.0-flash",
):
    console = Console()

    eval_task = EVAL_TASKS[eval_task_name]()
    model_wrapper = get_generation_wrapper(gen_source)

    simple_date = datetime.now().strftime("%m-%d-%-H-%-M")
    random_id = int(random.random() * 1000)
    run_name = f"{eval_task_name}_{gen_source}_{simple_date}_{random_id}"
    console.print(f"Starting eval run: {run_name}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "eval_results", run_name)
    ensure_directory(out_dir)

    eval_results: List[EvalResult] = []
    with Progress() as progress:
        dataset = eval_task.load_task_data()
        progress_bar_title = f"Eval task: {eval_task.name}"
        prog_task = progress.add_task(progress_bar_title, total=len(dataset))
        for batch in dataset.iter(batch_size=batch_size):
            batch_results = await eval_task.run_eval(batch, model_wrapper)  # type: ignore
            eval_results.extend(batch_results)

            _save_eval_results_to_md(eval_results, out_dir)
            _save_eval_results_to_csv(eval_results, out_dir)
            progress.advance(prog_task, batch_size)

    summary_dict = {}
    with open(f"{out_dir}/summary.json", "w") as f:
        f.write(json.dumps(summary_dict, indent=4))

    _save_eval_results_to_csv(eval_results, out_dir)
    _save_eval_results_to_md(eval_results, out_dir)


Fire(main)
