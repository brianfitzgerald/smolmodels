import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn.utils.rnn
from datasets import load_dataset
from loguru import logger
from peft.tuners.lora.config import LoraConfig
from peft.utils.constants import DUMMY_TARGET_MODULES
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import OptimizerNames
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from transformers.trainer_utils import SchedulerType

from dataset.code import (
    CodeContestsDataModule,
    UltraFeedbackDataModule,
)
from dataset.conversation import ConversationDataModule, ConversationDPODataModule
from dataset.playwright import PlaywrightSummaryToScript
from dataset.reasoning import GSM8KReasoningDataModule
from model.reasoning import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)
from model.utils import (
    DataModuleChoice,
    DatasetConfig,
    TuningModeChoice,
    ensure_directory,
    save_dataclass_to_json,
    short_hash,
)
from synthetic_data.utils import ldictl
from trl_wrapper.dpo_trainer import CustomDPOTrainer, EvalDataModeChoice
from trl_wrapper.sft_trainer import CustomSFTTrainer

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
    data_module_choice: DataModuleChoice = "ultra_feedback"
    dataset_path: Optional[str] = None
    custom_chat_template: Optional[str] = None
    eval_data_mode: EvalDataModeChoice = "random"
    # Max samples to use for training
    max_samples: Optional[int] = None
    max_eval_dataset_size: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000

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
    use_vllm: bool = False
    num_generations: int = 1


LLAMA_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_1B,
    max_samples=20000,
    n_epochs=10,
    data_module_choice="ultra_feedback",
)

DOLPHIN_DPO_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="dolphin-dpo",
    train_batch_size=12,
    max_samples=20000,
    using_mistral=True,
)

CODECONTESTS_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=12,
    data_module_choice="conversation_dpo",
    using_mistral=True,
)

CODECONTESTS_SFT_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=16,
    data_module_choice="conversation_dpo",
    using_mistral=True,
    tuning_mode="sft",
    learning_rate=1e-5,
)

GUTENBERG_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_3B,
    wandb_project_name="gutenberg",
    train_batch_size=8,
    gradient_accumulation_steps=4,
    data_module_choice="conversation",
    tuning_mode="sft",
    learning_rate=1e-5,
    dataset_path="dataset_files/gutenberg_conversations.parquet",
    n_epochs=5,
    eval_steps=50,
    run_suffix="gutenberg-conv",
)

GUTENBERG_DPO_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_3B,
    wandb_project_name="gutenberg",
    train_batch_size=4,
    gradient_accumulation_steps=2,
    data_module_choice="conversation_dpo",
    tuning_mode="dpo",
    learning_rate=1e-6,
    dataset_path="jondurbin/gutenberg-dpo-v0.1",
    n_epochs=5,
    run_suffix="gutenberg-dpo",
)


# llama 3 hparams
# https://huggingface.co/blog/llama3#fine-tuning-with-🤗-trl

CODECONTESTS_COT_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_3B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=4,
    data_module_choice="conversation",
    tuning_mode="sft",
    gradient_checkpointing=False,
    learning_rate=1e-5,
    n_epochs=10,
    train_on_inputs=False,
    special_tokens=["<thought>", "</thought>", "<solution>", "</solution>"],
    dataset_path="openo1_sft_formatted_thoughts_conversations.parquet",
    neftune_noise_alpha=5,
    lr_scheduler=SchedulerType.COSINE,
)

CODECONTESTS_DPO_CONFIG = WrapperConfig(
    model_id_or_path="01-09-17-46-208302-llama-3.2-3b-instruct-openo1-composite-1e-5",
    wandb_project_name="codecontests-dpo",
    train_batch_size=4,
    data_module_choice="conversation_dpo",
    dataset_path="jondurbin/py-dpo-v0.1",
    tuning_mode="dpo",
    gradient_checkpointing=False,
    learning_rate=1e-5,
    n_epochs=1,
)

ULTRAFEEDBACK_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_3B,
    wandb_project_name="ultrafeedback-dpo",
    train_batch_size=4,
    data_module_choice="ultra_feedback",
    tuning_mode="dpo",
    gradient_checkpointing=False,
    learning_rate=1e-5,
    n_epochs=1,
    max_samples=25000,
)

GRPO_MATH_CONFIG = WrapperConfig(
    model_id_or_path=SMOL_LM_135M,
    wandb_project_name="qwen-math-grpo",
    train_batch_size=4,
    gradient_accumulation_steps=4,
    data_module_choice="gsm8k_reasoning",
    max_prompt_length=256,
    max_grad_norm=0.1,
    eval_batch_size=1,
    learning_rate=5e-6,
    lr_scheduler=SchedulerType.COSINE,
    tuning_mode="grpo",
    use_vllm=False,
    num_generations=4,
)

CHAT_TEMPLATE_OVERRIDES = {
    "llama3": {LLAMA_3_1_8B, LLAMA_3_2_3B, LLAMA_3_2_1B},
    "smollmv2": {SMOL_LM_135M},
    "ministral": {MINISTRAL_8B},
}


CONFIGS = {
    "llama": LLAMA_CONFIG,
    "dolphin": DOLPHIN_DPO_CONFIG,
    "codecontests": CODECONTESTS_CONFIG,
    "codecontests_sft": CODECONTESTS_SFT_CONFIG,
    "codecontests_cot_sft": CODECONTESTS_COT_CONFIG,
    "codecontests_cot_dpo": CODECONTESTS_COT_CONFIG,
    "ultrafeedback": ULTRAFEEDBACK_CONFIG,
    "gutenberg": GUTENBERG_CONFIG,
    "gutenberg_dpo": GUTENBERG_DPO_CONFIG,
    "grpo_math": GRPO_MATH_CONFIG,
}

LOCAL_RUNS_FOLDER = "./runs"
MODELS_FOLDER = "/models"


class TrainerWrapper:
    def __init__(self, config: WrapperConfig, use_wandb: bool = False) -> None:
        self.config = config
        self.use_wandb = use_wandb

        # Init tokenizer here so we can use it without loading the model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id_or_path
        )  # type: ignore
        # https://github.com/huggingface/trl/issues/1311#issuecomment-2016614091
        # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if self.config.special_tokens is not None:
            logger.info(f"Adding special tokens: {self.config.special_tokens}")
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.config.special_tokens}  # type: ignore
            )  # type: ignore

    def init_model(self):
        bnb_config = None

        if self.config.tuning_mode == "lora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.using_mps = torch.mps.is_available()
        attn_impl = "sdpa" if self.using_mps else "flash_attention_2"
        logger.info(
            f"Loading model {self.config.model_id_or_path} with attn_impl: {attn_impl}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id_or_path,
            device_map="auto",
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,
            use_cache=not self.config.using_mistral,
        )
        if self.config.special_tokens is not None:
            logger.info(f"Resizing token embeddings for model to {len(self.tokenizer)}")
            # Cannot use mean_resizing as `torch.linalg.eigvals` is not supported on CUDA 12.4
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

    def init_data_module(self, dataset_root_path: str = ""):
        dataset_path = self.config.dataset_path
        if dataset_path is not None and not os.path.isabs(dataset_path):
            dataset_path = os.path.join(dataset_root_path, dataset_path)
            logger.info(f"Using dataset path: {dataset_path}")

        for k, v in CHAT_TEMPLATE_OVERRIDES.items():
            if self.config.model_id_or_path in v:
                logger.info(f"Using chat template override: {k}")
                self.config.custom_chat_template = k

        dataset_config = DatasetConfig(
            input_dataset_name=dataset_path,
            batch_size=self.config.train_batch_size,
            max_sequence_length=self.config.max_sequence_length,
            using_mistral=self.config.using_mistral,
            notebook_mode=self.config.notebook_mode,
            tuning_mode=self.config.tuning_mode,
            max_samples=self.config.max_samples,
            custom_chat_template=self.config.custom_chat_template,
            train_on_inputs=self.config.train_on_inputs,
        )
        # TODO clean this up
        if self.config.data_module_choice == "code_contests":
            self.data_module = CodeContestsDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "conversation":
            self.data_module = ConversationDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "ultra_feedback":
            self.data_module = UltraFeedbackDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "conversation_dpo":
            self.data_module = ConversationDPODataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "playwright_summary_to_script":
            self.data_module = PlaywrightSummaryToScript(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "gsm8k_reasoning":
            self.data_module = GSM8KReasoningDataModule(self.tokenizer, dataset_config)
        self.data_module.setup("fit")

    def init_trainer(self, config_name: str | None = None):
        # Get run name
        simple_date = datetime.now().strftime("%m-%d-%-H-%-M")
        random_id = int(torch.rand(1) * 1000000)
        model_id_without_org = self.config.model_id_or_path.split("/")[-1].lower()
        if not config_name:
            config_name = ""
        run_name = f"{simple_date}-{random_id}-{model_id_without_org}-{config_name}"
        if self.config.run_suffix is not None:
            run_name += f"-{self.config.run_suffix}"

        runs_folder = LOCAL_RUNS_FOLDER

        if os.path.exists(MODELS_FOLDER):
            runs_folder = os.path.join(MODELS_FOLDER, "runs")
            os.makedirs(MODELS_FOLDER, exist_ok=True)

        output_dir = os.path.join(runs_folder, run_name)
        logger.info(f"Saving output to: {output_dir}")

        os.environ["WANDB_PROJECT"] = self.config.wandb_project_name

        # LoRA config
        peft_config = None
        if self.config.use_lora:
            peft_config = LoraConfig(
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                r=self.config.lora_rank,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            if self.model is not None and hasattr(self.model, "peft_config"):
                logger.info("LoRA already loaded, ignoring")
                peft_config.target_modules = DUMMY_TARGET_MODULES

        ensure_directory(output_dir)
        save_dataclass_to_json(self.config, f"{output_dir}/wrapper_config.json")
        model_id_hash = short_hash(self.config.model_id_or_path)
        assert self.data_module is not None, "Data module not initialized"
        logger.info(
            f"Initializing trainer, run_name: {run_name}, wandb project: {self.config.wandb_project_name}"
        )
        if self.config.tuning_mode == "dpo":
            self.ref_logpbrobs_cache_location = (
                f"{self.data_module.cache_dir}/{model_id_hash}/ref_logprobs_cache"
            )
            logger.info(
                f"logprobs cache location: {self.ref_logpbrobs_cache_location} peft config: {peft_config is not None}"
            )

        # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L143
        if self.config.tuning_mode in ("sft", "sft_lora"):
            args = SFTConfig(
                num_train_epochs=self.config.n_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim=self.config.optimizer,
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                warmup_ratio=0.1,
                lr_scheduler_type=self.config.lr_scheduler.value,
                logging_steps=10,
                save_steps=self.config.save_steps,
                save_total_limit=2,
                eval_strategy="steps",
                eval_on_start=not self.config.notebook_mode,
                eval_steps=self.config.eval_steps,
                bf16=True,
                push_to_hub=False,
                report_to="wandb" if self.use_wandb else "none",
                dataloader_num_workers=0,
                dataset_num_proc=4,
                max_seq_length=self.config.max_sequence_length,
                dataloader_pin_memory=True,
                run_name=run_name,
                output_dir=output_dir,
                disable_tqdm=not self.config.notebook_mode,
                neftune_noise_alpha=self.config.neftune_noise_alpha,
                use_liger=True,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
            )

            def basic_pad_collator(examples):
                """
                need to pad labels separately for some stupid reason, main collator doesn't do this
                """
                padded = self.tokenizer.pad(
                    examples,
                )
                padded["labels"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.LongTensor(x) for x in ldictl(examples)["labels"]],
                    padding_value=self.tokenizer.pad_token_id,  # type: ignore
                ).transpose(0, 1)
                padded["input_ids"] = torch.LongTensor(padded["input_ids"])  # type: ignore
                padded["attention_mask"] = torch.LongTensor(padded["attention_mask"])
                if "assistant_mask" in examples:
                    padded["assistant_mask"] = torch.LongTensor(
                        padded["assistant_mask"]
                    )
                return padded

            self.trainer = CustomSFTTrainer(
                self.model,
                peft_config=peft_config,
                args=args,
                train_dataset=self.data_module.train_dataset,
                eval_dataset=self.data_module.val_dataset,
                tokenizer=self.tokenizer,  # type: ignore
                data_collator=basic_pad_collator,
            )
            self.trainer.set_custom_args(
                self.config.max_eval_sample_length,
                True,
                output_dir,
                self.config.eval_data_mode,
                self.config.using_mistral,
            )

        elif self.config.tuning_mode == "grpo":
            training_args = GRPOConfig(
                output_dir=output_dir,
                run_name=run_name,
                learning_rate=self.config.learning_rate,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.1,
                warmup_ratio=0.1,
                lr_scheduler_type=self.config.lr_scheduler.value,
                logging_steps=1,
                bf16=True,
                per_device_train_batch_size=self.config.train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_generations=self.config.num_generations,
                max_prompt_length=self.config.max_prompt_length,
                max_completion_length=self.config.max_completion_length,
                num_train_epochs=1,
                save_steps=self.config.save_steps,
                max_grad_norm=self.config.max_grad_norm,
                per_device_eval_batch_size=self.config.eval_batch_size,
                use_vllm=self.config.use_vllm,
                vllm_gpu_memory_utilization=0.3,
                vllm_device="cuda:0",
                report_to="wandb" if self.use_wandb else "none",
            )
            self.trainer = GRPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.data_module.train_dataset,
                eval_dataset=self.data_module.val_dataset,
                reward_funcs=[
                    xmlcount_reward_func,
                    soft_format_reward_func,
                    strict_format_reward_func,
                    int_reward_func,
                    correctness_reward_func,
                ],  # type: ignore
            )

        else:
            args = DPOConfig(
                num_train_epochs=self.config.n_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim=self.config.optimizer,
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                logging_steps=1,
                save_steps=self.config.save_steps,
                save_total_limit=2,
                eval_strategy="steps",
                eval_on_start=True,
                eval_steps=self.config.eval_steps,
                bf16=True,
                tf32=False,
                push_to_hub=False,
                report_to="wandb" if self.use_wandb else "none",
                dataloader_num_workers=0 if self.config.notebook_mode else 4,
                dataset_num_proc=1 if self.config.notebook_mode else 4,
                max_length=self.config.max_sequence_length,
                max_prompt_length=self.config.max_prompt_length,
                precompute_ref_log_probs=not self.config.using_mistral,
                precompute_ref_batch_size=self.config.logprob_precompute_batch_size,
                dataloader_pin_memory=True,
                beta=self.config.dpo_beta,
                loss_type="sigmoid",
                generate_during_eval=True,
                run_name=run_name,
                output_dir=output_dir,
                disable_tqdm=not self.config.notebook_mode,
            )

            self.trainer = CustomDPOTrainer(
                self.model,
                ref_model=None,  # set to none since we use peft
                peft_config=peft_config,
                args=args,
                train_dataset=self.data_module.train_dataset,
                eval_dataset=self.data_module.val_dataset,
                tokenizer=self.tokenizer,  # type: ignore
            )
            self.trainer.set_custom_args(
                self.config.max_eval_sample_length,
                True,
                output_dir,
                self.config.eval_data_mode,
                self.config.using_mistral,
            )

            if (
                self.trainer.precompute_ref_log_probs
                and self.config.tuning_mode == "dpo"
            ):
                eval_cache_location, train_cache_location = (
                    f"{self.ref_logpbrobs_cache_location}_eval.parquet",
                    f"{self.ref_logpbrobs_cache_location}_train.parquet",
                )
                if os.path.exists(train_cache_location):
                    logger.info("Loading cached logprobs...")
                    # TODO add support for eval dataset
                    self.trainer.train_dataset = load_dataset(
                        "parquet", data_files={"train": train_cache_location}
                    )["train"]  # type: ignore
                    self.trainer.eval_dataset = load_dataset(
                        "parquet", data_files={"train": eval_cache_location}
                    )["train"]  # type: ignore
                    self.trainer._precomputed_train_ref_log_probs = True
                    self.trainer._precomputed_eval_ref_log_probs = True
                    logger.info("Loaded.")
                else:
                    # force precomputing of reference logprobs
                    logger.info(
                        f"Precomputing reference logprobs, batch size: {self.config.logprob_precompute_batch_size}"
                    )
                    self.trainer.args.per_device_train_batch_size = (
                        self.config.logprob_precompute_batch_size
                    )

                    logger.info("Precomputing train logprobs")
                    self.trainer.get_train_dataloader()
                    logger.info("Precomputing eval logprobs")
                    self.trainer.get_eval_dataloader()
                    logger.info("Saving reference logprobs...")
                    assert self.trainer.eval_dataset is not None
                    assert self.trainer.train_dataset is not None

                    self.trainer.train_dataset.to_parquet(train_cache_location)
                    self.trainer.eval_dataset.to_parquet(eval_cache_location)
                    self.trainer.args.per_device_train_batch_size = (
                        self.config.train_batch_size
                    )

    def train(self):
        logger.info("Starting training.")
        self.trainer.train()
