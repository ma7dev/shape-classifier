from typing import Optional
import os
import random
import string
import yaml
import fire
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from shape_classifier.models.simple_classifier import Model
from shape_classifier.loss import FocalLoss
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, data_len, learning_rate,chkpt_path=""):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(num_classes)
        self.learning_rate = learning_rate
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss()
        self.metric = torchmetrics.Accuracy()
        self.data_len = data_len
        self.current_step = 1
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        # optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01,momentum=0.5)
        # warmup_factor = 1.0 / 1000
        # warmup_iters = min(1000, self.data_len - 1)
        # def fun(iter_num: int) -> float:
        #     if iter_num >= warmup_iters:
        #         return 1
        #     alpha = float(iter_num) / warmup_iters
        #     return warmup_factor * (1 - alpha) + alpha
        # lr_scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, fun),
        #     'name': 'learning_rate'
        # }
        # return [optimizer], [lr_scheduler]
        return optimizer
    
    def forward(self,x):
        logits = self.model(x)
        return logits
    def _step(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        # print(logits)
        # print(y)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)
        return {'loss': loss, 'acc': acc}
    def _epoch_end(self, outputs, stage=None):
        if stage:
            acc = np.mean([float(x['acc']) for x in outputs])
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/acc": acc}, self.current_epoch + 1)
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1)
    
    # on_train_start()
    # for epoch in epochs:
    #   on_train_epoch_start()
    #   for batch in train_dataloader():
    #       on_train_batch_start()
    #       training_step()
    #       ...
    #       on_train_batch_end()
    #       on_validation_epoch_start()
    #
    #       for batch in val_dataloader():
    #           on_validation_batch_start()
    #           validation_step()
    #           on_validation_batch_end()
    #       on_validation_epoch_end()
    #
    #   on_train_epoch_end()
    # on_train_end
    def training_step(self, batch, batch_idx):
        self.current_step += 1
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)