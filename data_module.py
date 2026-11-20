from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torchvision.transforms.v2 as transforms
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        num_classes: int,
        train_subset_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        dataloader_num_workers: int,
        # skip any model parameters
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset = dataset_dir
        self.num_classes = num_classes

        self.train_subset_size = train_subset_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.num_workers = dataloader_num_workers

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        train_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2430, 0.2610]
                ),
            ]
        )
        dataset = CIFAR10(
            root=self.dataset, train=True, transform=train_transform, download=True
        )
        subset_indices = list(range(self.train_subset_size))
        self.train_dataset = Subset(dataset, subset_indices)

        valid_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2430, 0.2610]
                ),
            ]
        )
        self.val_dataset = CIFAR10(
            root=self.dataset, train=False, transform=valid_transform, download=True
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )


class DPLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
    ):
        super().__init__()
        self.datamodule = datamodule

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return DPDataLoader.from_data_loader(dataloader, distributed=False)

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return self.datamodule.transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)
