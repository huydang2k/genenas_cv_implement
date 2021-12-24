from argparse import ArgumentParser
from typing import Optional
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from network.nasgep_cell_net import NasgepCellNet
from util.logger import ChromosomeLogger
from typing import List
from argparse import ArgumentParser

class NasgepNetRWE_multiObj(pl.LightningModule):
    
    def __init__(
        self,
        num_labels: int,
        dropout: float = 0.1,
        hidden_shape: List = [3,32,32],
        N: int = 1,
        input_size: int = 32,
        num_val_dataloader: int = 1,
        **kwargs,
    ):
        # print('init NasgepNetRWE')
        super().__init__()
        self.input_size = input_size
        self.hidden_shape = hidden_shape
        self.N = N
        self.save_hyperparameters()
        padding_for_conv_3x3 = (self.hidden_shape[1] - self.input_size + 1,self.hidden_shape[2] - self.input_size + 1)
        post_nasgep_cell_output_channel = self.hidden_shape[0]
        
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader
        self.chromosome_logger: Optional[ChromosomeLogger] = None
        self.metric = None
      
        self.temp_output = None
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=self.hidden_shape[0], kernel_size=3,padding = padding_for_conv_3x3) 
        )
        self.cls_head = nn.Sequential(
            nn.BatchNorm2d(post_nasgep_cell_output_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool2d(kernel_size = int(self.hidden_shape[1]/4) ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_shape[0], num_labels),
        )
    
    def init_model(self, cells, adfs):
        nasgepcell_net = NasgepCellNet(
            cells,
            adfs,
            self.hidden_shape,
            self.N
        )
        for param in nasgepcell_net.parameters():
            param.requires_grad = False
       
        self.add_module("nasgep_cell_net", nasgepcell_net)
    
    def total_params(self):
        return sum(p.numel() for p in self.nasgep_cell_net.parameters())
    
    def forward(self, feature_map, mode = 'validate'):
        if mode == 'inference':            
            x = self.embed(feature_map)
            x = self.nasgep_cell_net(x)
            x = self.cls_head(x)
            return x
        
        if mode == 'pre_calculate':   
            # print('pre_calculating')
            try:
                x = self.embed(feature_map.cuda())
            except:
                x = self.embed(feature_map.unsqueeze(0).cuda())
            x = self.nasgep_cell_net(x)
            return x.squeeze()
        
        if mode == 'validate':
            x = self.cls_head(feature_map)
            return x
            
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        embed = self.embed
        model = self.nasgep_cell_net
        cls = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                
                "params": 
                # do not train normal & reduction cell
                # [
                #     p
                #     for n, p in model.named_parameters()
                #     if not any(nd in n for nd in no_decay)
                # ]
                # +
                [
                    p
                    for n, p in cls.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                # + [
                #     p
                #     for n, p in embed.named_parameters()
                #     if not any(nd in n for nd in no_decay)
                # ]
                ,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": 
                #[
                #     p
                #     for n, p in model.named_parameters()
                #     if any(nd in n for nd in no_decay)
                # ]
                # + 
                [
                    p
                    for n, p in cls.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                # + [
                #     p
                #     for n, p in embed.named_parameters()
                #     if any(nd in n for nd in no_decay)
                # ]
                ,
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.epsilon,
        )

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
        return optimizer
    
    def reset_weights(self):
        self.cls_head.reset_parameters()
    
    def validation_step(self, batch, batch_idx):
       
        logits = self(batch[0]['feature_map'])
        labels = batch[1]['labels']
        onehot_labels = batch[1]['one_hot']
        val_loss = F.cross_entropy(logits, onehot_labels)
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        # labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}
    
    def training_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
        # No multiple eval_splits
        # Looking at you MNLI
        
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        
        # self.print(np.unique(preds, return_counts=True))
        # self.print(np.unique(labels, return_counts=True))
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        log_data = {f"val_loss": loss, "metrics": metrics, "epoch": self.current_epoch}
        self.chromosome_logger.log_epoch(log_data)
      
        

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
        parser.add_argument("--N", default=1, type=int)
        parser.add_argument("--hidden_shape", default= [3,32,32], nargs='+', type=int)
        # parser.add_argument("--input_size", default= 32, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        return parser
    
    def total_params(self):
        return sum(p.numel() for p in self.nasgep_cell_net.parameters())
    
    def init_chromosome_logger(self, logger: ChromosomeLogger):
        self.chromosome_logger = logger

    def init_metric(self, metric):
        self.metric = metric
    
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
        parser.add_argument("--N", default=1, type=int)
        parser.add_argument("--hidden_shape", default= [3,32,32], nargs='+', type=int)
        # parser.add_argument("--input_size", default= 32, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        return parser
        
class NasgepNet_multiObj(NasgepNetRWE_multiObj):
    def __init__(self, num_labels: int, dropout: float = 0.1, hidden_shape: List = [3, 32, 32], N: int = 1, input_size: int = 32, num_val_dataloader: int = 1, **kwargs):
        padding_for_conv_3x3 = (self.hidden_shape[1] - self.input_size + 1,self.hidden_shape[2] - self.input_size + 1)
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=int(self.hidden_shape[0]/2), kernel_size=3,padding = padding_for_conv_3x3,stride = 2) ,
            nn.Conv2d(in_channels= int(self.hidden_shape[0]/2), out_channels=self.hidden_shape[0], kernel_size=3,padding = (1,1),stride = 2) ,
            nn.Relu(),
            nn.Conv2d(in_channels= self.hidden_shape[0], out_channels=self.hidden_shape[0], kernel_size=3,padding = (1,1),stride = 2) ,
            nn.Relu(),
        )
        super().__init__(num_labels, dropout=dropout, hidden_shape=hidden_shape, N=N, input_size=input_size, num_val_dataloader=num_val_dataloader, **kwargs)
        
    def init_model(self, cells, adfs):
        nasgepcell_net = NasgepCellNet(
            cells,
            adfs,
            self.hidden_shape,
            self.N
        )
        self.add_module("nasgepcell_net", nasgepcell_net)
    
    def configure_optimizers(self):
        embed = self.embed
        model = self.nasgepcell_net
        cls = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                
                "params": 
               
                [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                +
                [
                    p
                    for n, p in cls.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in embed.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                ,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": 
                [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + 
                [
                    p
                    for n, p in cls.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in embed.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                ,
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.epsilon,
        )

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
        return optimizer
    
    def forward(self, feature_map):
        x = self.embed(feature_map)
        x = self.nasgepcell_net(x)
        x = self.cls_head(x)
        return x
    
    def validation_epoch_end(self, outputs):
        # No multiple eval_splits
        # Looking at you MNLI
        
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        
        # self.print(np.unique(preds, return_counts=True))
        # self.print(np.unique(labels, return_counts=True))
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        log_data = {f"val_loss": loss, "metrics": metrics, "epoch": self.current_epoch}
        self.chromosome_logger.log_epoch(log_data)
        #hardcode 
        acc = metrics['accuracy']
        
        print(f'epoch: {self.current_epoch}, val_loss: {loss}, accuracy: {acc} ')