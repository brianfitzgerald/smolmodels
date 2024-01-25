import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
from torch import Tensor
import lightning as L
import torch
import fire
from utils import (
    check_valid_checkpoint_dir,
)
from transformers import AutoTokenizer
from scripts.download import download_from_hub
from scripts.convert_hf_checkpoint import convert_hf_checkpoint

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from model import GPT, Config
from chat import (
    extract_text_from_generated_message,
    model_conversation_input,
)


@torch.inference_mode()
def generate(
    model: GPT,
    encoded_prompt: Tensor,
    max_returned_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    T = encoded_prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    device, dtype = encoded_prompt.device, encoded_prompt.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = encoded_prompt
    encoded_prompt = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for i in tqdm(range(max_returned_tokens - T)):
        x = encoded_prompt.index_select(0, input_pos).view(1, -1)

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
        encoded_prompt = encoded_prompt.index_copy(0, input_pos, next_token_idx)

        # if <eos> token is triggered, return the output (stop generation)
        # if the token was generated in the first few indices, then continue
        if next_token_idx == eos_id:
            return encoded_prompt[:input_pos]  # include the EOS token

    return encoded_prompt


def main(
    model_name: str = "rocket-3B",
    max_new_tokens: int = 1024,
    top_k: int = 64,
    temperature: float = 0.8,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """

    config: Config = Config.from_name(model_name)

    checkpoint_dir = f"checkpoints/{config.organization}/{config.name}"
    checkpoint_dir_path = Path(checkpoint_dir)

    model_file = "lit_model.pth"
    model_checkpoint_path = checkpoint_dir_path / model_file

    if not check_valid_checkpoint_dir(checkpoint_dir_path):
        download_from_hub(config.hf_path)
        convert_hf_checkpoint(checkpoint_dir, config)

    fabric = L.Fabric(devices=1, precision="bf16-true")
    fabric.launch()

    device = fabric.device
    print(f"Using device: {str(device)}")

    t0 = time.perf_counter()

    print(f"Instantiating model...")
    t0 = time.perf_counter()
    with fabric.init_module():
        model: GPT = GPT(config, device)
    model = fabric.setup_module(model)  # type: ignore
    model.eval()
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    print(f"Loading model weights...")
    t0 = time.perf_counter()
    state_dict = lazy_load(model_checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=True)
    print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.")

    message_history: List[Dict] = []

    tokenizer = AutoTokenizer.from_pretrained(config.hf_path, trust_remote_code=True)

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1, device=device)

    while True:
        user_prompt = input("Enter a message: ")
        message_history.append({"role": "user", "content": user_prompt})

        print(f"Message history:\n{message_history}")
        full_formatted_prompt = model_conversation_input(message_history)

        full_formatted_prompt_str = "\n".join(full_formatted_prompt)

        prompt = """<|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
        """

        system = "You are a helpful assistant."
        user = "How are you?"

        # Apply the ChatML format
        prompt = prompt.format(system=system, user=user)

        encoded: Tensor = tokenizer.encode(prompt, return_attention_mask=False, return_tensors="pt")[0].to(device)  # type: ignore
        encoded_context_length = encoded.shape[0]
        max_returned_tokens = encoded_context_length + max_new_tokens

        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_returned_tokens,
            temperature,
            top_k,
            tokenizer.eos_token_id,
        )
        t = time.perf_counter() - t0

        new_model_output = tokenizer.decode(y[encoded_context_length:])
        print(
            f"encoded_context_length: {encoded_context_length} total generation length: {y.size(0)}"
        )
        new_model_output = extract_text_from_generated_message(new_model_output)
        print(f"New output:\n{new_model_output}")
        message_history.append({"role": "assistant", "content": new_model_output})

        num_tokens_generated = y.size(0) - encoded_context_length
        print(
            f"Time for inference: {t:.02f} sec total, {num_tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )
        print(
            f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
            file=sys.stderr,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    fire.Fire(main)
