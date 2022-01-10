from sklearn.utils import validation
import torch 
import argparse
from problem import CV_DataModule_train
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class RESNET(pl.LightningModule):
    def __init__(self,
                 lr, 
                 eps,
                 wd,):
        self.lr = lr
        self.eps = eps
        self.wd = wd
        super(RESNET, self).__init__()
        self.model = ResNet(ResidualBlock, [2, 2, 2])
    
    def forward(self, x):
        return self.model(x)
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch[0]['feature_map'])
        labels = batch[1]['labels']
        onehot_labels = batch[1]['one_hot']
        val_loss = F.cross_entropy(logits, onehot_labels)
        preds = torch.argmax(logits, dim=1)
     
        return {"loss": val_loss, "preds": preds, "labels": labels}
    
    def training_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
    
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        metrics = accuracy_score(labels, preds)
        print('Epochs {}: val_loss: {}, accuracy: {}'.format(self.current_epoch, loss, metrics))
    
    def training_epoch_end(self, outputs):
        if (self.current_epoch+1) % 20 == 0:
            self.lr /= 3
            update_lr(self.optimizer, self.lr)
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        metrics = accuracy_score(labels, preds)
        print('Epochs {}: train_loss: {}, accuracy: {}'.format(self.current_epoch, loss, metrics))
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr = self.lr,
            eps = self.eps,
            weight_decay= self.wd,
        )

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
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