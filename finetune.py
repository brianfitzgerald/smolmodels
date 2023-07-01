from dataclasses import dataclass
from enum import Enum, IntEnum
from torch.utils.data import DataLoader
import evaluate
import torch
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import torch.nn as nn
import wandb
import os
import torch.nn.functional as F
from pprint import pprint


from transformers import (
    DataCollatorWithPadding,
    Trainer,
    get_scheduler,
    AutoTokenizer,
    AdamW,
    GPTNeoForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset

from utils import get_available_device, download_if_not_present
from peft import prepare_model_for_kbit_training

wandb.login()
wandb_run = wandb.init(project="smolmodels-finetune-mpt")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


ds_location = "./roleplay_dataset.json"


class DatasetChoice(IntEnum):
    CRD = 1
    ROLEPLAY_INSTRUCT = 2


@dataclass
class TrainingArgs:
    ds_choice = DatasetChoice.ROLEPLAY_INSTRUCT
    num_epochs = 6
    save_interval = 16
    eval_interval = 16


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

if TrainingArgs.ds_choice == DatasetChoice.ROLEPLAY_INSTRUCT:
    download_if_not_present(
        ds_location,
        "https://raw.githubusercontent.com/teknium1/GPTeacher/main/Roleplay/roleplay-simple-deduped-roleplay-instruct.json",
    )
    dataset = load_dataset("json", data_files=ds_location, split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    def merge_strings(x):
        merged = x["instruction"] + x["input"] + x["response"]
        return {"text": merged}

    dataset = dataset.map(merge_strings)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], padding=True), batched=True, batch_size=64
    )

    dataset = dataset.remove_columns(["instruction", "input", "response", "text"])
elif TrainingArgs.ds_choice == DatasetChoice.CRD:
    dataset = load_dataset("roborovski/crd-preproc", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True), batched=True, batch_size=64
    )

    dataset = dataset.remove_columns(
        [
            "turn_start",
            "turn_end",
            "chunk_id",
            "chunk",
            "turns",
            "text",
            "alignment_score",
        ]
    )


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["k_proj", "v_proj", "q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = get_available_device()

model = GPTNeoForCausalLM.from_pretrained(
    "roneneldan/TinyStories-Instruct-33M",
    quantization_config=bnb_config,
    device_map={"": 0},
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
optimizer = AdamW(model.parameters(), lr=5e-5)

wandb.watch(model)

print_trainable_parameters(model)


collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch_size = 16

train_dataloader = DataLoader(
    dataset["train"], batch_size=batch_size, collate_fn=collator
)
eval_dataloader = DataLoader(
    dataset["test"], batch_size=batch_size, collate_fn=collator
)

num_training_steps = TrainingArgs.num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

loss_fn = nn.CrossEntropyLoss()

progress_bar = tqdm(range(num_training_steps))
metric = evaluate.load("glue", "mrpc")


def get_perplexity(logits: torch.Tensor, input_ids: torch.Tensor):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perp = torch.exp(loss)
    return perp.item()


def get_text_sample(
    logits: torch.Tensor, input_ids: torch.Tensor, tokenizer: AutoTokenizer
):
    decoded_input = tokenizer.decode(input_ids[0])
    next_token_id = torch.argmax(logits, dim=1).item()
    decoded_next_token = tokenizer.decode(next_token_id)
    return decoded_input, decoded_next_token


model.train()

for epoch in range(TrainingArgs.num_epochs):
    for j, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        logits = outputs.loss["logits"].view(-1, tokenizer.vocab_size)
        input_ids = batch["input_ids"].view(-1)
        loss = loss_fn(logits, input_ids)

        loss.backward()

        loss_val = loss.item()

        wandb.log({"loss": loss})

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if j % TrainingArgs.eval_interval:
            model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                logits = outputs.loss["logits"].view(-1, tokenizer.vocab_size)
                input_ids = batch["input_ids"].view(-1)

                prompt, generated_text = get_text_sample(logits, input_ids)
                perplexity_score = get_perplexity(logits, input_ids)
                log_dict = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "perplexity_score": perplexity_score,
                }
                pprint(log_dict, index=2)
                wandb.log(log_dict)
            model.train()

        if j % TrainingArgs.save_interval:
            save_file_path = os.path.join(
                "checkpoints", f"model_epoch_{epoch}_batch_{j}.pt"
            )
            model.save(save_file_path)
