import os
import torch
from scripts.download import download_from_hub
from typing import Dict
from typing import Optional, Dict
from tqdm import tqdm
from torch import Tensor
import torch
from transformers import AutoTokenizer
import fire
from utils import get_available_device
import json
from models.t5 import T5, remap_state_dict


def main():
    device = get_available_device()

    model_org = "google"
    model_name = "t5-efficient-base"
    hf_path = f"{model_org}/{model_name}"

    download_from_hub(hf_path)

    config = json.load(
        open(os.path.join("checkpoints", model_org, model_name, "config.json"))
    )

    model = T5(
        dim=config["d_model"],
        enc_n_positions=config["n_positions"],
        vocab_size=config["vocab_size"],
        num_encoder_layers=config["num_layers"],
        enc_heads=config["num_heads"],
        enc_dim_head=config["d_kv"],
        enc_mlp_mult=4,
        num_decoder_layers=config["num_decoder_layers"],
        dec_heads=config["num_heads"],
        dec_dim_head=config["d_kv"],
        dec_mlp_mult=4,
        dropout=0.1,
        tie_token_emb=True,
    )
    model.to(device)

    print("Loading state dict")
    state_dict: Dict[str, torch.Tensor] = torch.load(
        os.path.join("checkpoints", model_org, model_name, "pytorch_model.bin")
    )

    state_dict = remap_state_dict(state_dict)

    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    prompt = "summarize: studies have shown that owning a dog is good for you"
    encoded_prompt: Tensor = tokenizer.encode(prompt, return_attention_mask=False, return_tensors="pt")[0].to(device)  # type: ignore
    encoded_prompt = encoded_prompt.unsqueeze(0)
    out_tensor = torch.randint(0, 512, (1, 1024), device=device)
    output = model(encoded_prompt, out_tensor).to(device)  # type: ignore
    breakpoint()
    decoded_output = tokenizer.decode(output[0])
    print(decoded_output)


if __name__ == "__main__":
    fire.Fire(main)
