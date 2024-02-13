print("Loading dependencies...")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

from model.utils import TASK_PREFIX
import pandas as pd
from fire import Fire


def main(
    checkpoint_dir: str = "checkpoints/best_model-v1.ckpt.dir",
    batch_size: int = 8,
    upload_to_hf: bool = False,
):

    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        checkpoint_dir
    )

    drawbench_df: pd.DataFrame = pd.read_csv("data/drawbench.csv")

    if upload_to_hf:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(file_dir, ".env"))
        token = os.getenv("HF_TOKEN")
        print(f"Logging in with token: {token}")
        login(token=token, add_to_git_credential=True)

        print("Uploading to HuggingFace...")
        model.push_to_hub("superprompt-v1")
        return

    for i in range(0, len(drawbench_df), batch_size):

        chunk = drawbench_df[i : i + batch_size]

        prompts_with_prefix = [TASK_PREFIX + sentence for sentence in chunk["Prompt"]]

        inputs = tokenizer(prompts_with_prefix, return_tensors="pt", padding=True)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_return_sequences=1,
        )

        out = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        for prompt, output in zip(chunk["Prompt"], out):
            print(f"Prompt: {prompt}\nOutput: {output}\n\n")


if __name__ == "__main__":
    Fire(main)
