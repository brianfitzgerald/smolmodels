from tokenizers import AutoTokenizer
from torch.utils.data import DataLoader
import evaluate
import torch

from transformers import (
    GenerationMixin,
    DataCollatorForLanguageModeling,
    get_scheduler,
    AdamW,
    GPTNeoForCausalLM,
)
from datasets import load_dataset
from utils import get_available_device


dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

tokenized_datasets = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        return_tensors="pt",
    ),
    batched=True,
)
tokenized_datasets = dataset.remove_columns(["text"])

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


model = GPTNeoForCausalLM()
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
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
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
