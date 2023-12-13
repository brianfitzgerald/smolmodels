from t5 import T5, remap_state_dict
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


@torch.inference_mode()
def generate(
    model: T5,
    input_ids: Tensor,
    max_returned_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    T = input_ids.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    device, dtype = input_ids.device, input_ids.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = input_ids
    input_ids = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for i in tqdm(range(max_returned_tokens - T)):
        x = input_ids.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        input_ids = input_ids.index_copy(0, input_pos, next_token_idx)

        # if <eos> token is triggered, return the output (stop generation)
        # if the token was generated in the first few indices, then continue
        if next_token_idx == eos_id:
            return input_ids[:input_pos]  # include the EOS token

    return input_ids


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
    out_tensor = torch.arange(encoded_prompt.size(0))
    output = model(encoded_prompt, out_tensor).to(device)  # type: ignore
    decoded_output = tokenizer.decode(output[0])
    print(decoded_output)


if __name__ == "__main__":
    fire.Fire(main)
