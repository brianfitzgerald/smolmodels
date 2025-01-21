import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from trl.trainer.dpo_trainer import PreferenceCollator

from synthetic_data.utils import dictl
from trl.trainer.dpo_trainer import DPOTrainer


class CustomTrainer:
    def __init__(self, trainer: DPOTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer

    def compute_loss_metrics(self, batch_size: int = 1):
        """
        Iterate through all batches and compute metrics sample-wise.
        Keep on batch_size=1 unless needed
        """
        assert self.trainer._peft_has_been_casted_to_bf16
        # precompute reference logprobs
        self.trainer.get_train_dataloader()
        collator = PreferenceCollator(self.tokenizer.pad_token_id, "pt")  # type: ignore
        outputs = []
        with torch.no_grad(), autocast("cuda"):
            batch_iter = tqdm(
                self.trainer.train_dataset.iter(batch_size=batch_size),
                desc="Computing DPO loss",
                total=len(self.trainer.train_dataset) // batch_size,
            )
            for batch in batch_iter:
                sample_collated = collator(dictl(batch))
                metrics = self.get_sample_wise_metrics(sample_collated)
                for j in range(batch_size):
                    out_sample = {
                        "prompt": batch["prompt"][j],
                        "chosen": batch["chosen"][j],
                        "rejected": batch["rejected"][j],
                    }
                    for k, v in metrics.items():
                        if isinstance(v, list):
                            out_sample[k] = v[j]
                        else:
                            out_sample[k] = v
                    outputs.append(out_sample)
        return outputs

    def get_sample_wise_metrics(self, batch: dict):
        """
        Return sample-wise loss metrics for a single batch
        """
        metrics = {}

        model_output = self.trainer.concatenated_forward(self.model, batch)  # type: ignore

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.trainer.compute_ref_log_probs(
                batch
            )

        losses, chosen_rewards, rejected_rewards = self.trainer.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

        metrics = {
            "loss": losses.tolist(),
            "reward_accuracy": reward_accuracies.tolist(),
            "reward_margin": reward_margins.tolist(),
            "chosen_rewards": chosen_rewards.tolist(),
            "rejected_rewards": rejected_rewards.tolist(),
        }

        for k in [
            "chosen_logps",
            "rejected_logps",
            "mean_chosen_logits",
            "mean_rejected_logits",
        ]:
            metrics[k] = model_output[k].tolist()
        return metrics
