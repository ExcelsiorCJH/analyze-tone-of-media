import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import OrderedDict
from utils.types_ import *

# Device configuration
GPU_NUM = 1
DEVICE = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Experiment(pl.LightningModule):
    def __init__(self, model: ClassType, params: Dict):
        super(Experiment, self).__init__()

        self.model = model
        self.params = params
        self._loss = nn.CrossEntropyLoss()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, sequences: Tensor) -> Tensor:
        return self.model(sequences)[0]

    def loss_function(self, preds: Tensor, labels: Tensor) -> Tensor:
        ce_loss = self._loss(preds, labels)
        return ce_loss

    def accuracy(self, preds: Tensor, labels: Tensor) -> Tensor:
        preds = torch.max(preds, 1)[1]
        corrects = (preds == labels).float().sum()
        acc = corrects / labels.numel()
        return acc

    def training_step(self, batch, batch_idx):
        sequences, labels, keywords = batch

        preds = self.forward(sequences)
        train_loss = self.loss_function(preds, labels)
        train_acc = self.accuracy(preds, labels)
        tqdm_dict = {"train_acc": train_acc}

        output = OrderedDict({"loss": train_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def validation_step(self, batch, batch_idx):
        sequences, labels, keywords = batch

        preds = self.forward(sequences)
        val_loss = self.loss_function(preds, labels)
        val_acc = self.accuracy(preds, labels)

        output = OrderedDict({"val_loss": val_loss, "val_acc": val_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": val_loss_mean, "val_acc": val_acc_mean}

    def test_step(self, batch, batch_idx):
        sequences, labels, keywords = batch

        preds = self.forward(sequences)
        test_loss = self.loss_function(preds, labels)
        test_acc = self.accuracy(preds, labels)

        output = OrderedDict({"test_loss": test_loss, "test_acc": test_acc,})
        return output

    def test_epoc_end(self, outputs):
        val_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": val_loss_mean}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params["LR"], weight_decay=1e-5)

