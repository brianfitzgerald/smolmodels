from typing import Optional
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from huggingface_hub import login, HfApi
import os
from dotenv import load_dotenv
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch
from PIL import Image
from synthetic_data.utils import print_result_dicts, ensure_directory


import pandas as pd
from fire import Fire

from train import CONFIGS


def format_filename(string: str, limit: int = 20):
    result = string.lower().replace(" ", "_")
    result = "".join(char if char.isalnum() or char == "_" else "" for char in result)
    result = result[:limit]
    return result


def main(
    checkpoint_file: str = "checkpoints/prompt_safety-zCN6",
    dataset_file: Optional[str] = None,
    config: str = "prompt_safety",
    batch_size: int = 8,
    upload_to_hf: bool = False,
    generate_samples: bool = False,
    sdxl: bool = True,
):

    model_config = CONFIGS[config]
    loaded_checkpoint = torch.load(checkpoint_file)
    model_state_dict = {
        key.replace("model.", ""): value
        for key, value in loaded_checkpoint["state_dict"].items()
    }

    tokenizer = T5Tokenizer.from_pretrained(
        model_config.hyperparams.base_model_checkpoint
    )
    hf_config = T5Config.from_pretrained(model_config.hyperparams.base_model_checkpoint)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        None, config=hf_config, state_dict=model_state_dict
    )
    model = model.to("cuda")  # type: ignore

    if generate_samples:
        pipeline_name = (
            "stabilityai/stable-diffusion-xl-base-1.0"
            if sdxl
            else "runwayml/stable-diffusion-v1-5"
        )
        print(f"Loading pipeline: {pipeline_name}")
        pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            pipeline_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe = pipe.to("cuda")

    if upload_to_hf:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(file_dir, ".env"))
        token = os.getenv("HF_TOKEN")
        print(f"Logging in with token: {token}")
        login(token=token, add_to_git_credential=True)
        hf_api = HfApi(token=token)
        print("Uploading model card...")
        hf_api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id="roborovski/superprompt-v1",
            repo_type="model",
        )

        print("Uploading model...")
        model.push_to_hub("superprompt-v1")
        return

    out_dir = f"samples/{model_config.ckpt_name}"
    ensure_directory(out_dir, clear=True)

    if dataset_file:
        print(f"Reading samples from: {dataset_file}")
        if dataset_file.endswith(".csv"):
            validation_df: pd.DataFrame = pd.read_csv(dataset_file)
            validation_df.rename(columns={"Prompt": "prompt"}, inplace=True)
        elif dataset_file.endswith(".parquet"):
            validation_df: pd.DataFrame = pd.read_parquet(dataset_file)
            validation_df = validation_df[
                (validation_df["nsfw_regex"] == True)
                | (validation_df["nsfw_image"] == True)
                | len(validation_df["prompt"]) < 64
            ]

        # TODO refactor this so the parquet is loaded as a dataset, and uses the same dataloader
        for i in range(0, len(validation_df), batch_size):

            chunk = validation_df[i : i + batch_size]

            prompts_with_prefix = [
                model_config.task_prefix + sentence for sentence in chunk["prompt"]
            ]

            inputs = tokenizer(
                prompts_with_prefix, return_tensors="pt", padding=True
            ).to("cuda")

            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=77,
                num_return_sequences=1,
            )

            generated_strings = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            new_prompt_rows = []

            for j, (prompt, generated) in enumerate(
                zip(chunk["prompt"], generated_strings)
            ):
                new_prompt_rows.append({"Prompt": prompt, "Sanitized": generated})
                if generate_samples:
                    for k, txt in enumerate([prompt, generated]):
                        print(f"Generating sample for: {txt}")
                        image: Image.Image = pipe(txt, num_inference_steps=30, guidance_scale=20).images[0]  # type: ignore
                        prompt_fmt = format_filename(txt)
                        label = "prompt" if k == 0 else "upsampled"
                        image.save(f"{out_dir}/{i}_{j}_{k}_{label}_{prompt_fmt}_.png")

            print_result_dicts(new_prompt_rows)
    else:
        print("Loading samples from dataloader")
        data_module = model_config.data_module(
            batch_size, tokenizer, model_config.hyperparams.max_seq_length
        )
        data_module.setup("validate")
        rows = []
        for i, inputs in enumerate(data_module.val_dataloader()):
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")
            labels = inputs["labels"]
            print(f"Generating {len(input_ids)} completions...")
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=77,
                num_return_sequences=1,
            )
            generated_strings = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            prompt_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for prompt, generated in zip(prompt_strings, generated_strings):
                rows.append({"Prompt": prompt, "Sanitized": generated})

            print_result_dicts(rows[-batch_size:])

            if i % 100 == 0 and i > 0:
                print(f"Processed {i} samples, saving...")
                out_df = pd.DataFrame(rows)
                out_df.to_csv(f"{out_dir}/saferprompt_out.csv", index=False)


if __name__ == "__main__":
    Fire(main)
