from argparse import ArgumentParser
import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, train_test_split
from typing import Tuple
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import linear
from torchvision import models, transforms
import torchvision
from torch.utils.data import DataLoader, random_split
from torch import optim 
from collections import defaultdict
import copy
class CV_DataModule(pl.LightningDataModule):
    metrics_names = {
        "cifar10": "accuracy",
        # "health_fact": "accuracy",
    }
    num_labels_map = {
        "cifar10": 10
    }
    dataset_names = {
        "cifar10": "CIFAR10"
        # "health_fact": ["health_fact"],
    }
    def __init__(
        self,
        task_name: str,
        input_shape : Tuple[int,int,int] = [3,32,32],
        input_size:int = 32,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        cache_dataset: bool = False,
        cached_dataset_filepath: str = "",
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.input_shape = input_shape
        self.input_size = input_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.cached_train = None
        self.cached_vals = None
        self.cache_dataset = cache_dataset
        if self.cache_dataset:
            if not cached_dataset_filepath:
                cached_dataset_filepath = f"{self.task_name}.cached.dataset.pt"
            self.load_cache_dataset(cached_dataset_filepath)
        self.num_labels = self.num_labels_map[self.task_name]
    def convert_img(self,img):
        return {'feature_map':self.transform(img)}
    def setup(self, stage):
        if not self.cache_dataset:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.dataset = {}
            self.dataset['train'] = getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./data', 
                train=True, 
                download=True,
                transform=self.convert_img)

            self.dataset['test'] = getattr(torchvision.datasets, self.task_name.upper())(root='./data',
                train=False,
                download=True,
                transform=self.convert_img
                )
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
        else:
            if self.task_name in ["cifar10"]:
                self.dataset["test"] = self.dataset["validation"]
            split_dict = self.dataset["train"].train_test_split(test_size=0.1, seed=42)
            self.dataset["train"] = split_dict["train"]
            self.dataset["validation"] = split_dict["test"]
            
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
      

    def prepare_data(self):
        if not self.cache_dataset:
            getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./data',download=True)
       

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True,num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, shuffle=True,num_workers=2)
    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, shuffle=True,num_workers=2)

    @property
    def train_dataset(self):
        return self.dataset["train"]

    @property
    def val_dataset(self):
        # if len(self.eval_splits) == 1:
        return self.dataset["validation"]
        # elif len(self.eval_splits) > 1:
            # return [self.dataset[x] for x in self.eval_splits]

    @property
    def metric(self):
        return datasets.load_metric(self.metrics_names[self.task_name])

 
    def kfold(self, k_folds=10, seed=420):
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.train_dataset)):
            train_ids = train_ids.tolist()
            val_ids = val_ids.tolist()
            
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            
            yield fold, DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                sampler=train_subsampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ), DataLoader(
                self.train_dataset,
                batch_size=self.eval_batch_size,
                sampler=val_subsampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),

    def load_cache_dataset(self, cached_dataset_filepath):
        print(f"Load cached dataset {cached_dataset_filepath}")
        self.dataset = torch.load(cached_dataset_filepath)

    @staticmethod
    def add_cache_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--cache-dataset",
            action="store_true",
            help="If use cached dataset",
        )
        parser.add_argument(
            "--cache-dataset-filepath", type=str, default="", help="Cached dataset path"
        )
        parser.add_argument("--k-folds", type=int, default=10)

        return parser
