from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from model.utils import TASK_PREFIX
import pandas as pd


def main(checkpoint_path: str):

    drawbench = pd.read_csv("data/drawbench.csv")

    prompts_with_prefix = [TASK_PREFIX + sentence for sentence in drawbench["Prompt"]]

    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        checkpoint_path
    )

    inputs = tokenizer(prompts_with_prefix, return_tensors="pt", padding=True)

    output_sequences = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
