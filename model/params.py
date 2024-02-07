
class HyperParams:
    model_name: str = "google/flan-t5-base"
    max_seq_length: int = 256
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    n_gpus: int = 1
    fp_16: bool = False
    max_grad_norm: float = 10.0
    seed: int = 42
    weight_decay: float = 0.0

