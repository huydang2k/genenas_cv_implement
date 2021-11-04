from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Union, Dict

import numpy as np
import datasets
import pytorch_lightning as pl
import torch
import torch.nn as nn

from cv_config import cv_problem_config
from network.nasgep_cell_net import NasgepCellNet
from util.logger import ChromosomeLogger
from util.exception import NanException

class NasgepNet(pl.LightningModule):
    def __init__(
        
        self,
        num_labels: int,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        num_val_dataloader: int = 1,
        use_simple_cls: bool = True,
        **kwargs,
    ):
        print('init NasgepNet')
        super().__init__()
        self.input_size = cv_problem_config['image_size']
        self.N = cv_problem_config['N']
        self.save_hyperparameters()
        #
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=cv_problem_config['first_conv_3x3_number_kernel'], kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(cv_problem_config['first_conv_3x3_number_kernel']  / 4)
        self.relu = nn.ReLU()
        feature_map_dim = cv_problem_config['image_size'] - 3 + 1
        self.global_avg_pool = nn.AvgPool2d(feature_map_dim)
        
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader
        self.cell_dropout = nn.Dropout(p=dropout)

        hidden_size = cv_problem_config['first_conv_3x3_number_kernel']  / 4
        if not use_simple_cls:
            self.cls_head = ClsHead(hidden_size, dropout, num_labels)
        else:
            self.cls_head = SimpleClsHead(hidden_size, dropout, num_labels)

        self.chromosome_logger: Optional[ChromosomeLogger] = None
        self.metric = None

    def init_metric(self, metric):
        self.metric = metric
    
    #add  module Nasgep Cell
    def init_model(self, cells, adfs):
        nasgepcell_net = NasgepCellNet(
            cells,
            adfs,
            self.config.hidden_size,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=self.hparams.batch_first,
            bidirection=self.hparams.bidirection,
        )
        for param in nasgepcell_net.parameters():
            param.requires_grad = False
        self.add_module("nasgepcell_net", nasgepcell_net)

    def reset_weights(self):
        self.cls_head.reset_parameters()

    def validation_epoch_end(self, outputs):
        # No multiple eval_splits
        # Looking at you MNLI
        import numpy as np

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        self.print(np.unique(preds, return_counts=True))
        self.print(np.unique(labels, return_counts=True))
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        log_data = {f"val_loss": loss, "metrics": metrics, "epoch": self.current_epoch}
        self.chromosome_logger.log_epoch(log_data)

    def init_chromosome_logger(self, logger: ChromosomeLogger):
        self.chromosome_logger = logger

    def forward(self, hiddens, **inputs):
        print('Foward Net')
        print('input ', inputs)
        print('hidden ',hiddens)
        print('Hidden shape: ',hiddens.shape)
        labels = None
        if "labels" in inputs:
            labels = inputs.pop("labels")
        x = self.embed(**inputs)[0]
        # if x.isnan().any():
        #     raise NanException(f"NaN after embeds")
        x, hiddens = self.nasgepcell_net(x, hiddens)
        
        # if x.isnan().any():
        #     raise NanException(f"NaN after recurrent")
        if self.hparams.batch_first:
            x = x[:, 0, :]  # CLS token
        else:
            x = x[0, :, :]
        logits = self.cls_head(x)
        # if logits.isnan().any():
        #     raise NanException(f"NaN after CLS head")
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, hiddens

    def training_step(self, batch, batch_idx, hiddens=None):
        loss, _, hiddens = self(hiddens, **batch)
        return {"loss": loss, "hiddens": hiddens}

    def tbptt_split_batch(self, batch, split_size):
        num_splits = None
        split_dict = {}
        for k, v in batch.items():
            if k == "labels":
                split_dict[k] = v
                continue
            else:
                split_dict[k] = torch.split(
                    v, split_size, int(self.hparams.batch_first)
                )
                assert (
                    num_splits == len(split_dict[k]) or num_splits is None
                ), "mismatched splits"
                num_splits = len(split_dict[k])

        new_batch = []
        for i in range(num_splits):
            batch_dict = {}
            for k, v in split_dict.items():
                if k == "labels":
                    batch_dict[k] = v
                else:
                    batch_dict[k] = v[i]
            new_batch.append(batch_dict)

        return new_batch

    def validation_step(self, batch, batch_idx):
        val_loss, logits, _ = self(None, **batch)

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (
                    len(train_loader.dataset)
                    // (self.hparams.train_batch_size * max(1, self.hparams.gpus))
                )
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        embed = self.embed
        model = self.recurrent_model
        fc = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in embed.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in embed.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def total_params(self):
        return sum(p.numel() for p in self.recurrent_model.parameters())

    def reset_weights(self):
        self.cls_head.reset_parameters()
        self.recurrent_model.reset_parameters()

    @staticmethod
    def add_learning_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--batch_first", default=True, type=bool)
        parser.add_argument("--bidirection", default=True, type=bool)
        parser.add_argument("--hidden_size", default=128, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--unfreeze_embed", action="store_false")
        parser.add_argument("--use_simple_cls", action="store_true")
        return parser


class SimpleClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x, **kwargs):
        x = torch.tanh(x)
        x = self.dense(x)
        return x

    def reset_parameters(self):
        self.dense.reset_parameters()
