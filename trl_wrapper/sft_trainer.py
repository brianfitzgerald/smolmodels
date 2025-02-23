import random
from typing import List, Optional, Dict, Tuple
import torch

import wandb
from loguru import logger
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalLoopOutput
from trl import SFTTrainer
from transformers.generation.configuration_utils import GenerationConfig
from trl.trainer.utils import pad_to_length
from datasets import Dataset

from model.utils import IGNORE_TOKEN_INDEX, PAD_TOKEN_ID
from synthetic_data.utils import EvalDataModeChoice, clean_message, log_to_file


class CustomSFTTrainer(SFTTrainer):
    def set_custom_args(
        self,
        max_eval_sample_length: int,
        eval_skip_special_tokens: bool,
        output_dir: str,
        eval_data_mode: EvalDataModeChoice,
        using_mistral: bool,
    ):
        """
        Set custom arguments needed for new functionality.
        This is so we don't have to modify the __init__ method of the Trainer class.
        """
        self.all_eval_rows = []
        self.max_eval_sample_length = max_eval_sample_length
        self.eval_skip_special_tokens = eval_skip_special_tokens
        self.output_dir = output_dir
        self.eval_data_mode = eval_data_mode
        self.using_mistral = using_mistral
        self.generate_during_eval = True

    def generate_from_model(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[str]:
        """Generate samples from the model."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.

        generation_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=self.max_eval_sample_length,
            max_time=20,
            # https://github.com/huggingface/trl/issues/1217
            use_cache=not self.using_mistral,
        )

        pad_token_id: int = self.processing_class.pad_token_id  # type: ignore

        with torch.autocast("cuda"):
            logger.info(
                f"Generating policy samples, max length: {self.max_eval_sample_length}..."
            )
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
            )

        model_output = pad_to_length(
            model_output, self.max_eval_sample_length, pad_token_id
        )
        model_output_decoded = self.processing_class.batch_decode(  # type: ignore
            model_output, skip_special_tokens=self.eval_skip_special_tokens
        )

        return model_output_decoded

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            batch = next(iter(dataloader))
            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch = self._prepare_inputs(batch)

            logger.info("Generating samples...")
            input_ids, labels, assistant_mask = (
                random_batch["input_ids"],
                random_batch["labels"],
                torch.tensor(
                    random_batch["assistant_mask"],
                    device=random_batch["input_ids"].device,
                ),
            )
            input_ids = torch.where(assistant_mask == 1, PAD_TOKEN_ID, input_ids)
            prompt_decoded = self.processing_class.batch_decode(  # type: ignore
                input_ids,
                skip_special_tokens=self.eval_skip_special_tokens,
            )
            labels_decodeable = labels.clone()
            labels_decodeable[labels_decodeable == -100] = (
                self.processing_class.pad_token_id
            )
            labels_decoded = self.processing_class.batch_decode(  # type: ignore
                labels_decodeable,
                skip_special_tokens=self.eval_skip_special_tokens,
            )
            policy_output_decoded = self.generate_from_model(
                self.model, input_ids, attention_mask=batch["attention_mask"]
            )  # type: ignore
            logger.info("Generated samples.")

            new_rows_to_log = []

            for i in range(len(prompt_decoded)):
                new_row_dict = {
                    "prompt": prompt_decoded[i],
                    "ref": labels_decoded[i],
                    "policy": policy_output_decoded[i],
                }
                new_row_dict = {k: clean_message(v) for k, v in new_row_dict.items()}
                new_rows_to_log.append(new_row_dict)

            new_rows_wandb_format = [k.values() for k in new_rows_to_log]
            wandb_headers = list(new_rows_to_log[0].keys())

            # TODO log to tabulate table if not using wandb, and save to txt file
            self.log(
                {
                    "eval_samples": wandb.Table(
                        columns=wandb_headers,
                        rows=new_rows_wandb_format,
                    )  # type: ignore
                }
            )
            log_to_file(
                self.all_eval_rows,
                new_rows_to_log,
                self.output_dir,
                self.state.global_step,
            )

            self.all_eval_rows.extend(new_rows_to_log)

            self.state.log_history.pop()

        # Base evaluation
        initial_output = Trainer.evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output
