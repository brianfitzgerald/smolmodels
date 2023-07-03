from enum import IntEnum
from pprint import pprint
from torch.utils.data import DataLoader
import evaluate
import torch
from dataclasses import dataclass
from evaluate import load
import torch.nn.functional as F
import os

from transformers import (
    DataCollatorForLanguageModeling,
    GPTNeoConfig,
    get_scheduler,
    AutoTokenizer,
    AdamW,
    GPTNeoForCausalLM,
)
from datasets import load_dataset
from utils import *
from tqdm.auto import tqdm
import wandb


class Task(IntEnum):
    TINY_STORIES = 1
    STATE_CHANGES = 2


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/"


@dataclass
class TrainingArgs:
    task = Task.STATE_CHANGES
    use_wandb = False
    model_name = "smolmodels-finetune-33m-crd-instruct"


if TrainingArgs.use_wandb:
    wandb.login()
    wandb_run = wandb.init(project=TrainingArgs.model_name)


def calculate_bpc(model, evaluation_data):
    total_loss = 0.0
    total_characters = 0

    model.eval()

    with torch.no_grad():
        for input_seq, target_seq in evaluation_data:
            input_seq = torch.tensor(input_seq).unsqueeze(0)
            target_seq = torch.tensor(target_seq).unsqueeze(0)

            output_seq = model(input_seq)
            output_seq = output_seq.squeeze(0)

            loss = F.cross_entropy(output_seq, target_seq)
            total_loss += loss.item()
            total_characters += target_seq.size(1)

    average_loss = total_loss / total_characters
    bpc = average_loss / torch.log(torch.tensor(2.0))

    return bpc.item()


tokenize_fn = None

if TrainingArgs.task == Task.TINY_STORIES:
    dataset = load_dataset("roneneldan/TinyStories")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenize_fn = lambda x: tokenizer(x["text"])

elif TrainingArgs.task == Task.STATE_CHANGES:
    dataset = load_dataset("Fraser/python-state-changes")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    def tokenize_state_changes(batch: dict):
        concatted_batch = []
        for i in range(len(batch["start"])):
            concatted = (
                batch["start"][i]
                + tokenizer.sep_token
                + batch["code"][i]
                + tokenizer.sep_token
                + batch["end"][i]
                + tokenizer.sep_token
            )
            concatted_batch.append(concatted)

        tokenized = tokenizer(
            concatted_batch, padding="max_length", truncation=True, return_tensors="pt"
        )
        return tokenized

    tokenize_fn = tokenize_state_changes

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
)
tokenized_dataset = tokenized_dataset.remove_columns(["start", "code", "end"])
tokenized_dataset.set_format("torch")
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8)
eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=8)

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
optimizer = AdamW(model.parameters(), lr=5e-5)

if TrainingArgs.use_wandb:
    wandb.watch(model)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = get_available_device()
model.to(device)


progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    print("epoch", epoch, train_dataloader)
    for j, batch in enumerate(train_dataloader):
        print(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        wandb.log({"loss": loss.item()})

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        print(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        input_ids = batch["input_ids"]
        predictions = torch.argmax(logits, dim=-1)

        wandb.log({"loss": loss.item()})

        prompt, generated_text = get_text_sample(logits, input_ids, tokenizer)
        perplexity_score = get_perplexity(logits, input_ids)
        log_dict = {
            "prompt": prompt,
            "generated_text": generated_text,
            "perplexity_score": perplexity_score,
        }
        pprint(log_dict, indent=2)
        if TrainingArgs.use_wandb:
            wandb.log(log_dict)

    if j % TrainingArgs.save_interval == 0:
        save_file_path = os.path.join("checkpoints", f"model_epoch_{epoch}_batch_{j}")
        if TrainingArgs.use_peft:
            model.save_pretrained(save_file_path, safe_serialization=True)
        if TrainingArgs.push_model:
            model.push_to_hub(TrainingArgs.model_name)
