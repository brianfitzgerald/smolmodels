from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import (
    DatasetConfig,
    SmDataset,
)
from synthetic_data.utils import ldictl
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

MASK_IDX = -100


class PlaywrightSummaryToScript(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)
        self.collator = DataCollatorForCompletionOnlyLM(
            "<scene>", "<summary>", tokenizer=tokenizer
        )

    def process_samples_batch(self, examples: dict):
        out = []
        for summary, scene in zip(examples["summary"], examples["scene"]):
            tokenized = self.tokenizer(
                f"<summary>{summary}<scene>{scene}",
                padding="max_length",
            )
            out.append(tokenized)
        return self.collator(out)

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns(
            ["name", "scene", "summary"]
        )
        self.val_dataset = self.val_dataset.remove_columns(["name", "scene", "summary"])
