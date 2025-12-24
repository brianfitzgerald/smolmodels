"""TRL training wrapper for smolmodels - handles SFT, DPO, and GRPO training."""

from trl_wrapper.trainer_wrapper import TrainerWrapper, CONFIGS
from trl_wrapper.wrapper_config import WrapperConfig, DatasetConfig, SmDataset
from trl_wrapper.dpo_trainer import CustomDPOTrainer
from trl_wrapper.sft_trainer import CustomSFTTrainer

__all__ = [
    "TrainerWrapper",
    "CONFIGS",
    "WrapperConfig",
    "DatasetConfig",
    "SmDataset",
    "CustomDPOTrainer",
    "CustomSFTTrainer",
]
