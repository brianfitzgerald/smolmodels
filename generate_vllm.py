from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import fire
from vllm import LLM, SamplingParams
from dalle import get_dalle_model_input
from pathlib import Path

# https://github.com/vllm-project/vllm/pull/1235/files

def main(
    prompt: str = "a dog with a hat",
    model_name: str = "PY007/TinyLlama-1.1B-Chat-v0.3",
    force_quantize: bool = False,
):

    model_dir = Path(model_name)
    quantized_model_dir = "quantized" / model_dir
    quantized_model_dir.mkdir(parents=True, exist_ok=True)
    quantized_model_path_str = quantized_model_dir.as_posix()
    quantized_ckpt_path = quantized_model_dir / "pytorch_model.bin"

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
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(quantized_model_path_str)
        breakpoint()
        tokenizer.save_pretrained(quantized_model_path_str)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
    model = LLM(quantized_model_path_str, dtype="auto", quantization="awq")
    dalle_model_input = get_dalle_model_input(prompt, llama_tokenizer)  # type: ignore
    outputs = model.generate(dalle_model_input, sampling_params)
    outputs_text = outputs[0].outputs[0].text
    print(outputs_text)


if __name__ == "__main__":
    fire.Fire(main)
