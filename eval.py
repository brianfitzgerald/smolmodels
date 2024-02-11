print("Loading dependencies...")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from model.utils import TASK_PREFIX
import pandas as pd
from fire import Fire


def main(
    checkpoint_path: str = "",
    batch_size: int = 8,
):

    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        checkpoint_path
    )

    drawbench_df: pd.DataFrame = pd.read_csv("data/drawbench.csv")

    for i in range(0, len(drawbench_df), batch_size):

        chunk = drawbench_df[i : i + batch_size]

        prompts_with_prefix = [TASK_PREFIX + sentence for sentence in chunk["Prompt"]]

        inputs = tokenizer(prompts_with_prefix, return_tensors="pt", padding=True)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        out = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        print(out)

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))


if __name__ == "__main__":
    Fire(main)
