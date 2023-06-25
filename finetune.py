from torch.utils.data import DataLoader
import evaluate
import torch
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

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

download_if_not_present(
    ds_location,
    "https://raw.githubusercontent.com/teknium1/GPTeacher/main/Roleplay/roleplay-simple-deduped-roleplay-instruct.json",
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

dataset = load_dataset("json", data_files=ds_location)
print(dataset)


def tokenize_step(batch):
    batch = tokenizer(batch["instruction"], batch["response"])
    return batch


tokenized_dataset = dataset.map(
    tokenize_step,
    batched=True,
)
tokenized_dataset.set_format("torch")

config = LoraConfig(
    r=8,
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
    "roneneldan/TinyStories-Instruct-33M", quantization_config=bnb_config, device_map={"":0}
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(model)
model = get_peft_model(model, config)
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)

print_trainable_parameters(model)

metric = evaluate.load("glue", "mrpc")

model.train()
tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
