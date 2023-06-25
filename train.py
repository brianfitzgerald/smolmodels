from dataclasses import dataclass
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from tokenizers import AutoTokenizer

from model import *
from transformers import GenerationMixin, DataCollatorForLanguageModeling
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
config = GPTNeoConfig()

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_position_embeddings,
        return_tensors="pt",
    ),
    batched=True,
)
dataset = dataset.remove_columns(["text"])

gpt = GPTNeoForCausalLM(config)


prompt = "Once upon a time there was"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

sampler_mixin = GenerationMixin()

output = gpt.generate(input_ids, max_length=1000, num_beams=1)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
