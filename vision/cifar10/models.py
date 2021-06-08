"""
A collection of CIFAR-10 classifiers.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import accuracy


class LeNet(nn.Module):
    """A slightly modern LeNet5 architecture (expects 32 x 32 x 3 images)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    pass


class VGG(nn.Module):
    pass


class ResNet(nn.Module):
    pass


class LitClassifier(LightningModule):
    """Container for a general classifier."""

    def __init__(self, model: nn.Module, lr: float = 0.01):
        super().__init__()
        self.model = model
        self.criterion = F.cross_entropy
        self.metric = accuracy
        self.save_hyperparameters("lr")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._eval(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._eval(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._eval(batch, prefix="test")

    def configure_optimizers(self):
        """Configure optimizer and optionally a learning rate scheduler."""
        optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor="train_loss")

    def _eval(self, batch, prefix):
        """Run inference on a batch and log loss and metrics."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        metric = self.metric(F.softmax(y_hat, dim=1), y)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_metric", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @classmethod
    def add_argparse_args(cls, parser):
        """Add model specific args to an existing parser."""
        group = parser.add_argument_group(cls.__name__)
        group.add_argument("--lr", type=float, default=0.01, help="learning rate")
        return parser
