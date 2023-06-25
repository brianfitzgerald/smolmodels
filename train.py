from torch.utils.data import DataLoader
import evaluate
import torch

from transformers import (
    DataCollatorWithPadding,
    GPTNeoConfig,
    get_scheduler,
    AutoTokenizer,
    AdamW,
    GPTNeoForCausalLM,
)
from datasets import load_dataset
from utils import get_available_device

# Script for training the base model

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

tokenized_datasets = dataset.map(
    lambda x: tokenizer(x["text"]),
    batched=True,
)
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

config = GPTNeoConfig(
    hidden_size=768,
    embed_dropout=0,
    attention_dropout=0,
    resid_dropout=0,
    max_position_embeddings=2048,
    num_heads=12,
    num_layers=12,
    attention_types=[[["global", "local"], 6]],
    window_size=256,
    layer_norm_epsilon=1e-5,
)

model = GPTNeoForCausalLM(config)
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = get_available_device()
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
metric = evaluate.load("glue", "mrpc")

model.train()

for epoch in range(num_epochs):
    print("epoch", epoch, train_dataloader)
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        print("batch", batch)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
