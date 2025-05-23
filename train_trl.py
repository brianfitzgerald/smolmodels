import fire
from loguru import logger
from trl_wrapper.trainer_wrapper import (
    TrainerWrapper,
    CONFIGS,
)
from dotenv import load_dotenv


def main(
    config: str = "dolphin",
    notebook_mode: bool = False,
    wandb: bool = False,
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"
    assert config in CONFIGS, f"Unknown config: {config}"
    load_dotenv(".env")

    cfg = CONFIGS[config]
    wrapper = TrainerWrapper(cfg, notebook_mode, wandb)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer(config)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()


if __name__ == "__main__":
    fire.Fire(main)
