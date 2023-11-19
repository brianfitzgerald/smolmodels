from io import BytesIO
from awq import AutoAWQForCausalLM
from awq.models.base import BaseAWQForCausalLM
from transformers import AutoTokenizer
import fire
from vllm import LLM, SamplingParams
from chat import model_conversation_input, extract_text_from_generated_message
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
from diffusers import DiffusionPipeline, LCMScheduler # type: ignore
import torch
from PIL import Image

import discord
from discord.ext import commands
from dotenv import load_dotenv
import os


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
        file.write(json.dumps(model.quant_config.to_dict(), indent=4))

def main(
    # force re-quantization even if the quantized model exists
    force_quantize: bool = False,
):

    # Load token

    load_dotenv(".env")
    discord_token = os.getenv("DISCORD_TOKEN")
    assert discord_token

    # Model IDs

    diffusion_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    diffusion_lora_id = "latent-consistency/lcm-lora-sdxl"
    language_model_id: str = "HuggingFaceH4/zephyr-7b-beta"

    # Quantization

    model_dir = Path(language_model_id)
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
        vllm_model = AutoAWQForCausalLM.from_pretrained(
            language_model_id, **{"low_cpu_mem_usage": True}
        )
        tokenizer = AutoTokenizer.from_pretrained(language_model_id, trust_remote_code=True)
        # make sure to change the os.remove line
        vllm_model.quantize(tokenizer, quant_config=quant_config)
        save_quantized(vllm_model, quantized_model_path_str)
        tokenizer.save_pretrained(quantized_model_path_str)

    # Loading models

    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(language_model_id, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
    vllm_model = LLM(quantized_model_path_str, dtype="auto", quantization="awq")
    
    print("Loading diffusion model...")
    diffusion_pipeline = DiffusionPipeline.from_pretrained(diffusion_model_id, variant="fp16")

    diffusion_pipeline.load_lora_weights(diffusion_lora_id)
    diffusion_pipeline.scheduler = LCMScheduler.from_config(diffusion_pipeline.scheduler.config)
    diffusion_pipeline.to(device="cuda", dtype=torch.float16)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = commands.Bot(intents=intents, command_prefix=".")

    allowed_channels = set([1003462226172194928, 1012434244859080825])

    @bot.event
    async def on_ready():
        assert bot.user
        print("Logged in as")
        print(bot.user.name)
        print(bot.user.id)
        await bot.tree.sync()
        print("------")

    @bot.listen()
    async def on_message(message: discord.Message):
        print('get message', message.author.id, message.author.name, message.channel.id)
        assert bot.user
        if message.author.id == bot.user.id:
            return
        if message.channel.id not in allowed_channels:
            return
        model_input = model_conversation_input(message.content, [], prompt_enhancer_mode=False)
        t0 = time.perf_counter()
        outputs = vllm_model.generate(model_input, sampling_params)
        outputs_text = outputs[0].outputs[0].text
        outputs_text = extract_text_from_generated_message(outputs_text)
        generated_image: Image.Image = diffusion_pipeline(
            prompt=outputs_text,
            num_inference_steps=4,
            guidance_scale=1
        ).images[0] # type: ignore
        with BytesIO() as image_binary:
            generated_image.save(image_binary, 'PNG')
            image_binary.seek(0)
            await message.channel.send(outputs_text, file=discord.File(fp=image_binary, filename='image.png'))
        print(outputs_text)
        print(f"Time taken: {time.perf_counter() - t0:.2f}s")

    bot.run(discord_token)

if __name__ == "__main__":
    fire.Fire(main)
