import lightning.pytorch as pl
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import DataLoader
from model.utils import HyperParams


class FineTunerDataset(pl.LightningDataModule):
    def __init__(
        self,
        hyperparams: HyperParams,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = hyperparams.train_batch_size
        self.tokenizer = tokenizer
        self.max_token_length = hyperparams.max_seq_length
        self.cpu_count = 1

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_count)  # type: ignore
