"""
Train a CIFAR-10 classifier.
"""

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datamodules import CIFAR10DataModule
from models import LitClassifier
from models import LeNet
from models import AlexNet
from models import VGG
from models import ResNet


def init_model(name: str):
    """Return an instance of the given model."""
    if args.model == "LeNet":
        return LeNet()
    if args.model == "AlexNet":
        return AlexNet()
    if args.model == "VGG":
        return VGG()
    if args.model == "ResNet":
        return ResNet()


def train(args):
    """Train a CIFAR-10 classifier."""
    pl.seed_everything(9)

    model = init_model(args.model)
    model = LitClassifier(model, lr=args.lr)
    cifar10 = CIFAR10DataModule.from_argparse_args(args)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(project="CIFAR-10", name=args.model, config=args, offline=True)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=lr_monitor, logger=logger)
    trainer.fit(model, datamodule=cifar10)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_argparse_args(parser)
    parser = CIFAR10DataModule.add_argparse_args(parser)

    models = ("LeNet", "AlexNet", "VGG", "ResNet")
    parser.add_argument("--model", type=str, required=True, choices=models)

    args = parser.parse_args()
    train(args)
