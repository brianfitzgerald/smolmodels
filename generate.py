import sys
import time
from pathlib import Path
from typing import Literal, Optional, Dict, Tuple, List
from tqdm import tqdm
from torch import Tensor

import torch
import fire
from utils import (
    check_valid_checkpoint_dir,
    load_checkpoint,
    get_available_device,
    weight_sum,
)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import GPT, Config
from tokenizer import Tokenizer
from dalle import model_conversation_input
from transformers import AutoTokenizer
from prompt_toolkit import prompt, PromptSession


@torch.inference_mode()
def generate(
    model: GPT,
    encoded_prompt: Tensor,
    max_returned_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
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
        if next_token_idx == eos_id:
            return encoded_prompt[:input_pos]  # include the EOS token

    return encoded_prompt


def main(
    num_samples: int = 10,
    max_new_tokens: int = 64,
    top_k: int = 200,
    temperature: float = 0.2,
    checkpoint_dir: str = "PY007/TinyLlama-1.1B-Chat-v0.3",
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

    checkpoint_dir_path = Path(checkpoint_dir)
    model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir_path / model_file

    check_valid_checkpoint_dir(checkpoint_dir_path)

    config = Config.from_name(checkpoint_dir_path.name)

    device = get_available_device()
    print(f"Using device: {str(device)}")

    t0 = time.perf_counter()

    dtype = torch.float16
    print(f"Instantiating model...")
    t0 = time.perf_counter()
    model: GPT = GPT(config, device)
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    print(f"Loading model weights...")
    t0 = time.perf_counter()
    load_checkpoint(model, checkpoint_path)
    print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.")

    message_history: List[Dict] = []

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = Tokenizer(checkpoint_dir_path)

    model.set_kv_cache(batch_size=1, device=device)

    for i in range(num_samples):

        user_prompt = prompt("Enter your prompt or modification: ")
        message_history.append({"role": "user", "content": user_prompt})

        full_formatted_prompt = model_conversation_input(user_prompt, message_history, llama_tokenizer)  # type: ignore
        encoded = tokenizer.encode(full_formatted_prompt, device=device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        breakpoint()

        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_returned_tokens,
            temperature,
            top_k,
            tokenizer.eos_id,
        )
        t = time.perf_counter() - t0

        new_model_output = tokenizer.decode(y[prompt_length:])
        print(new_model_output)
        message_history.append({"role": "assistant", "content": new_model_output})

        num_tokens_generated = y.size(0) - prompt_length
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {num_tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )
        print(
            f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
            file=sys.stderr,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    fire.Fire(main)
