"""
A PyTorch Lighting DataModule for CIFAR-10.
"""

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    """Container for CIFAR10 train, val, and test dataloaders."""

    def __init__(self, data_dir: str = "./data", batch_size: int = 32, n_workers: int = 3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dims = (3, 32, 32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255 for x in (125.3, 123.0, 113.9)],
                std=[x / 255 for x in (63.0, 62.1, 66.7)],
            )
        ])

    def prepare_data(self):
        """Download CIFAR10 train and test sets."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Configure dataloaders from train, val, and test."""
        if stage == "fit" or stage is None:
            cifar10 = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_ds, self.val_ds = random_split(cifar10, (45000, 5000))
        if stage == "test" or stage is None:
            self.test_ds = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, num_workers=self.n_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=self.n_workers)
