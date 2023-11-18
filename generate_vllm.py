from awq import AutoAWQForCausalLM
from awq.models.base import BaseAWQForCausalLM
from transformers import AutoTokenizer
import fire
from vllm import LLM, SamplingParams
from chat import model_conversation_input
from pathlib import Path
import time
import torch.nn as nn
from safetensors.torch import save_file
import os
import os
import json
import torch.nn as nn
from safetensors.torch import save_file
from transformers.modeling_utils import shard_checkpoint


# https://github.com/vllm-project/vllm/pull/1235/files

def save_quantized(model: BaseAWQForCausalLM, save_dir: str, shard_size="10GB"):

    # Save model
    class EmptyModule(nn.Module):
        def __init__(self): super(EmptyModule, self).__init__()
        def forward(self, x): return x

    # Save model files with empty state dict
    model.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

    # model_name has no extension, add it when saving state_dict
    model_name = 'model.safetensors'

    # Remove empty state dict
    os.remove(f'{save_dir}/{model_name}')

    # shard checkpoint into chunks (10GB default)
    shards, index = shard_checkpoint(
        model.model.state_dict(), 
        max_shard_size=shard_size, 
        weights_name=model_name
    )

    for shard_file, shard in shards.items():
        # safetensors must be in the same memory, so we duplicate and use contiguous memory
        shard = {k: v.clone().contiguous() for k, v in shard.items()}
        save_file(shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"})

    # save shard index
    if index is not None:
        with open(f'{save_dir}/{model_name}.index.json', 'w+') as file:
            file.write(json.dumps(index, indent=4))

    # Save config
    with open(f'{save_dir}/quant_config.json', 'w+') as file:
        file.write(json.dumps(model.quant_config, indent=4))

def main(
    prompt: str = "a dog with a hat",
    model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    force_quantize: bool = False,
    n_samples = 10,
):

    model_dir = Path(model_name)
    quantized_model_dir = "quantized" / model_dir
    quantized_model_dir.mkdir(parents=True, exist_ok=True)
    quantized_model_path_str = quantized_model_dir.as_posix()
    quantized_ckpt_path = quantized_model_dir / "model.safetensors"

    if not quantized_ckpt_path.exists() or force_quantize:
        print("Quantizing model...")
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        model = AutoAWQForCausalLM.from_pretrained(
            model_name, **{"low_cpu_mem_usage": True}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # make sure to change the os.remove line
        model.quantize(tokenizer, quant_config=quant_config)
        save_quantized(model, quantized_model_path_str)
        tokenizer.save_pretrained(quantized_model_path_str)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Loading model...")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=64)
    model = LLM(quantized_model_path_str, dtype="auto", quantization="awq")
    dalle_model_input = model_conversation_input(prompt, tokenizer)  # type: ignore
    for _ in range(n_samples):
        t0 = time.perf_counter()
        outputs = model.generate(dalle_model_input, sampling_params)
        outputs_text = outputs[0].outputs[0].text
        print(outputs_text)
        print(f"Time taken: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    fire.Fire(main)
