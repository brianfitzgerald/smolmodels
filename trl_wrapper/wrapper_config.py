from typing import Literal, Optional, List
from dataset.code import (
    CodeContestsDataModule,
)
from dataset.conversation import (
    ConversationDataModule,
    ConversationDPODataModule,
)
from dataset.playwright import PlaywrightSummaryToScript
from model.reasoning import (
    GSM8KDataModule,
)
from model.utils import (
    DataModuleChoice,
    SmDataset,
    TuningModeChoice,
)
from dataclasses import dataclass
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import SchedulerType
from synthetic_data.utils import EvalDataModeChoice

MOCK_LLAMA = "qgallouedec/tiny-LlamaForCausalLM-3"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_2_3B_BASE = "meta-llama/Llama-3.2-3B"
QWEN_2_5_3B = "Qwen/Qwen2.5-3B-Instruct"
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
SMOL_LM_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
# NOTE that mistral doesn't allow using system prompts, so it must be set to None.
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MINISTRAL_8B = "mistralai/Ministral-8B-Instruct-2410"

QWEN_0_5_B = "Qwen/Qwen2.5-0.5B-Instruct"

DataCollatorChoice = Literal["basic", "chat"]

DATA_MODULE_MAP: dict[DataModuleChoice, type[SmDataset]] = {
    "code_contests": CodeContestsDataModule,
    "conversation": ConversationDataModule,
    "gsm8k": GSM8KDataModule,
    "conversation_dpo": ConversationDPODataModule,
    "playwright_summary_to_script": PlaywrightSummaryToScript,
}


@dataclass
class WrapperConfig:
    # Model & Adapter Configuration
    model_id_or_path: str = LLAMA_3_2_1B
    using_mistral: bool = False
    adapter_path: Optional[str] = None

    # Experiment / Environment Settings
    notebook_mode: bool = False
    wandb_project_name: str = "codecontests-llama-3b"
    run_suffix: Optional[str] = None
    special_tokens: Optional[List[str]] = None

    # Data & Evaluation Configuration
    data_module_choice: DataModuleChoice = "conversation"
    dataset_path: Optional[str] = None
    eval_data_mode: EvalDataModeChoice = "random"
    # Max samples to use for training
    max_samples: Optional[int] = None
    max_eval_dataset_size: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    data_collator_choice: DataCollatorChoice = "basic"

    # Prompt & Sequence Lengths
    max_sequence_length: int = 1512  # sequence length for trimming completions
    max_prompt_length: int = 1024
    max_eval_sample_length: int = 1024
    max_completion_length: int = 200

    # Training Parameters
    train_batch_size: int = 4
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    n_epochs: int = 1
    train_on_inputs: bool = False

    # Optimization & Scheduling
    learning_rate: float = 5e-5
    max_grad_norm: float = 0.3
    lr_scheduler: SchedulerType = SchedulerType.CONSTANT
    optimizer: str = OptimizerNames.ADAMW_8BIT.value
    neftune_noise_alpha: Optional[float] = None
    dpo_beta: float = 0.1

    # Tuning / LoRA Configuration
    tuning_mode: TuningModeChoice = "sft"
    use_lora: bool = False
    lora_rank: int = 256
    lora_dropout: float = 0.05
    lora_alpha: int = 128
    logprob_precompute_batch_size: int = 16

    # Generation Parameters
    num_generations: int = 1


@dataclass
class DatasetConfig(WrapperConfig):
    chat_template_path: str | None = None
