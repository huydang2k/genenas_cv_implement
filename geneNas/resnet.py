import torch 
import argparse
from problem import CV_DataModule_train
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

class RESNET(pl.LightningModule):
    def __init__(self,
                 lr, 
                 eps,
                 wd,):
        self.lr = lr
        self.eps = eps
        self.wd = wd
        super(RESNET, self).__init__()
        self.model = models.resnet18(pretrained= False)
        self.fc = nn.Linear(1000,10)
    
    def forward(self, x):
        return self.fc(self.model(x))
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch[0]['feature_map'])
        labels = batch[1]['labels']
        onehot_labels = batch[1]['one_hot']
        print(logits)
        print(logits.shape, onehot_labels.shape)
        val_loss = F.cross_entropy(logits, onehot_labels)
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr = self.lr,
            eps = self.eps,
            weight_decay= self.wd,
        )
    
    
class Resnet(pl.LightningModule):
    def __init__(self, args):
        super(Resnet, self).__init__()
        self.hprams = args
        self.model = RESNET(args.lr, args. eps, args.wd)
        self.dm = CV_DataModule_train.from_argparse_args(args)
        self.dm.setup("fit",input_size= args.input_size)
        self.if_train = args.if_train
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.eval_batch_size
        self.load_model_checkpoint = args.load_model_checkpoint
        self.epochs = args.epochs
    
    def setup_trainer(self):
        trainer = pl.Trainer.from_argparse_args(
            self.hparams,
            gpus =1,
            num_sanity_val_steps= 0,
            max_epochs = self.epochs,
        )
        return trainer
    
    def train(self):
        self.trainer = self.setup_trainer()
        self.model.train()
        train_dataloader = DataLoader(self.dm.dataset['train'], batch_size= self.train_batch_size, shuffle= True)
        val_dataloader = DataLoader(self.dm.dataset['validation'], batch_size= self.val_batch_size)
        self.trainer.fit(
            self.model, 
            train_dataloaders= train_dataloader,
            val_dataloaders= val_dataloader,
        )
        
    @staticmethod
    def add_train_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type= float, default= 1e-3)
        parser.add_argument("--eps", type= float, default= 1e-9)
        parser.add_argument("--wd", type= float, default= 1e-5)
        parser.add_argument("--load_model_checkpoint",action='store_true')
        parser.add_argument("--if_train",action='store_true')
        parser.add_argument("--epochs",type= int, default= 100)
        return parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CV_DataModule_train.add_argparse_args(parser)
    parser = CV_DataModule_train.add_cache_arguments(parser)
    parser = Resnet.add_train_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args
        
def main():
    args = parse_args()
    resnet = Resnet(args)
    resnet.train()

if __name__ == "__main__":
    main()