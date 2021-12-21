from argparse import ArgumentParser
import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
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
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np

def split_stratify(dataset: Dataset, test_size):
    _, valid_idx= train_test_split(
    np.arange(dataset.__len__()),
    test_size= test_size,
    shuffle=True,
    stratify= dataset.data['labels'])
    return torch.utils.data.Subset(dataset, valid_idx)
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
        train_percentage,
        task_name: str,
        input_shape : Tuple[int,int,int] = [3,32,32],
        input_size:int = 32,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        cache_dataset: bool = False,
        cache_dataset_filepath: str = "",
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

        self.train_percentage = train_percentage
        self.cached_train = None
        self.cached_vals = None
        self.cache_dataset = cache_dataset
        self.cached_dataset_filepath = cache_dataset_filepath
        if self.cache_dataset:
            if not cache_dataset_filepath:
                self.cached_dataset_filepath = f"{self.task_name}.cached.dataset.pt"
            
            # self.load_cache_dataset(cached_dataset_filepath)
        self.num_labels = self.num_labels_map[self.task_name]
        self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def convert_img(self,img):
        return {'feature_map':self.transform(img)}
    
    def setup(self, stage):
        
        if not self.cache_dataset:
            print('not cache')
            self.dataset = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./cifar10_data', 
                train=True, 
                download= True,
                transform=self.convert_img)), self.train_percentage)

            self.dataset['test'] = tensor_dataset(getattr(torchvision.datasets, self.task_name.upper())(root='./cifar10_data',
                train=False,

                transform=self.convert_img)
                )
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
        else:
            print('cache')
            print(self.cached_dataset_filepath)
            self.dataset = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root=self.cached_dataset_filepath, 
                transform=self.convert_img)), self.train_percentage)

            self.dataset['test'] = tensor_dataset(getattr(torchvision.datasets, self.task_name.upper())(root=self.cached_dataset_filepath,
                train=False,
                transform=self.convert_img)
                )
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
            
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
      

    def prepare_data(self):
        if not self.cache_dataset:
            getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./cifar10_data',download=True)
       

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True,num_workers= self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, shuffle=True,num_workers= self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, shuffle=True,num_workers= self.num_workers)

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
                num_workers= self.num_workers,
                pin_memory=self.pin_memory,
            ), DataLoader(
                self.train_dataset,
                batch_size=self.eval_batch_size,
                sampler=val_subsampler,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory,
            ),

    # def load_cache_dataset(self, cached_dataset_filepath):
    #     print(f"Load cached dataset {cached_dataset_filepath}")
    #     self.dataset = torch.load(cached_dataset_filepath)

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
        parser.add_argument("--train_percentage", type=float, default=0.1)
        return parser 
    
    
    
def one_hot_labels(y, num_labels):
    return {'one_hot': F.one_hot(torch.tensor(y), num_labels).type(torch.float), 'labels': y}  
class CV_DataModule_RWE(CV_DataModule):
    def __init__(
        self,
        train_percentage: float,
        task_name: str,
        input_shape : Tuple[int,int,int] = [3,32,32],
        input_size:int = 32,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        cache_dataset: bool = False,
        cache_dataset_filepath: str = "",
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__(
            train_percentage= train_percentage,
            task_name= task_name,
            input_shape= input_shape,
            input_size= input_size,
            train_batch_size= train_batch_size,
            eval_batch_size= eval_batch_size,
            cache_dataset= cache_dataset,
            cache_dataset_filepath= cache_dataset_filepath,
            pin_memory= pin_memory,
            num_workers= num_workers,
        )
    
    def setup_device(self, gpus:int =1):
        self.gpus = gpus
    
    def convert_img(self,img):
        return {'feature_map': self.transform(img)}
    
    def get_model(self, model: pl.LightningModule):
        self.model = model
    
    def setup(self, stage):
        if not self.cache_dataset:
            self.onehot = lambda x: one_hot_labels(x, self.num_labels)
            print('not cache')
            self.dataset = {}
            self.dataset_ga = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./cifar10_data', 
                train=True, 
                download=True,
                transform=self.convert_img,
                target_transform = self.onehot
                )), self.train_percentage)
            self.dataset['test'] = getattr(torchvision.datasets, self.task_name.upper())(root='./cifar10_data',
                train=False,
                download=True,
                transform=self.convert_img,
                target_transform = self.onehot
            ) 
            print('Precalculating')
            start = time.time()
            self.dataset_ga['train'] = split_stratify(precalculated_dataset(self.dataset['train'], self.model, self.eval_batch_size, self.gpus), self.train_percentage)
            self.dataset_ga['test'] = precalculated_dataset(self.dataset['test'], self.model, self.eval_batch_size, self.gpus)
            end = time.time()
            print('Finish precalculating, Time: ', end - start)
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
            self.dataset_ga['validation'] = copy.deepcopy(self.dataset_ga['test'])
        else:
            self.onehot = lambda x: one_hot_labels(x, self.num_labels)
            print('not cache')
            self.dataset = {}
            self.dataset_ga = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root=self.cached_dataset_filepath, 
                train=True, 
                transform=self.convert_img,
                target_transform = self.onehot
                )), self.train_percentage)
            self.dataset['test'] = getattr(torchvision.datasets, self.task_name.upper())(root=self.cached_dataset_filepath,
                train=False,
                transform=self.convert_img,
                target_transform = self.onehot
            ) 
            print('Precalculating')
            start = time.time()
            self.dataset_ga['train'] = split_stratify(precalculated_dataset(self.dataset['train'], self.model, self.eval_batch_size, gpus = self.gpus), self.train_percentage)
            self.dataset_ga['test'] = precalculated_dataset(self.dataset['test'], self.model, self.eval_batch_size, gpus = self.gpus)
            end = time.time()
            print('Finish precalculating, Time: ', end - start)
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
            self.dataset_ga['validation'] = copy.deepcopy(self.dataset_ga['test'])
            
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x] 

    def kfold(self, k_folds=10, seed=420):
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset_ga['train'])):
            train_ids = train_ids.tolist()
            val_ids = val_ids.tolist()
            
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            
            yield fold, DataLoader(
                self.dataset_ga['train'],
                batch_size=self.train_batch_size,
                sampler=train_subsampler,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory,
            ), DataLoader(
                self.dataset_ga['train'],
                batch_size=self.eval_batch_size,
                sampler=val_subsampler,
                num_workers= self.num_workers,
                pin_memory=self.pin_memory,
            ),

class tensor_dataset(Dataset):
    def __init__(self, dataset):
        self.data = {}
        self.combine(dataset= dataset)
    
    def combine(self, dataset):
        feature_map = []
        labels = []    
        for i in  range(len(dataset)):
            feature_map.append(dataset[i][0]['feature_map'])
            labels.append(dataset[i][1])
        self.data['feature_map'] = torch.stack(feature_map).to('cpu')
        self.data['labels'] = torch.tensor(labels).to('cpu')
    
    def __len__(self):
        return len(self.data['feature_map'])

    def __getitem__(self, index):
        return ({'feature_map': self.data['feature_map'][index]}, self.data['labels'][index])
    
class precalculated_dataset(Dataset):
    def __init__(self, dataset, model = None, batch_size = 2056, gpus:int= 1):
        self.data = {}
        if gpus> 0:
            model.cuda()
        else:
            model.cpu()
        self.precalculate(dataset, model, batch_size)
        

    def precalculate(self, dataset, model, batch_size):
        feature_map = []
        labels = []
        one_hot = []
        for i in  tqdm(range(0, len(dataset), batch_size)):
            end = min(i + batch_size, len(dataset))
            tensor = []
            for j in range(i, end):
                tensor.append(dataset[j][0]['feature_map'])
                labels.append(dataset[j][1]['labels'])
                one_hot.append(dataset[j][1]['one_hot'].unsqueeze(0))
            tensor = model(torch.stack(tensor), mode = 'pre_calculate').detach()
            feature_map.append(tensor)
            
        self.data['feature_map'] = torch.cat(feature_map).to('cpu')
        self.data['labels'] = torch.tensor(labels).to('cpu')
        self.data['one_hot'] = torch.cat(one_hot).to('cpu')
        
        
            
    
    def __len__(self):
        return len(self.data['feature_map'])

    def __getitem__(self, index):
        return ({'feature_map': self.data['feature_map'][index]}, {'labels': self.data['labels'][index], 'one_hot': self.data['one_hot'][index]})

class CV_DataModule_train(CV_DataModule):
    def __init__(self, 
                 train_percentage: float,
                 task_name: str, 
                 input_shape: Tuple[int, int, int] = [3, 32, 32], 
                 input_size: int = 32, 
                 train_batch_size: int = 32, 
                 eval_batch_size: int = 32, 
                 cache_dataset: bool = False, 
                 cache_dataset_filepath: str = "", 
                 num_workers: int = 4, 
                 pin_memory: bool = True, **kwargs):
        super().__init__(
                         train_percentage= train_percentage,
                         task_name= task_name, 
                         input_shape=input_shape, 
                         input_size=input_size, 
                         train_batch_size=train_batch_size, 
                         eval_batch_size=eval_batch_size, 
                         cache_dataset=cache_dataset, 
                         cache_dataset_filepath=cache_dataset_filepath, 
                         num_workers=num_workers, 
                         pin_memory=pin_memory, 
                         **kwargs)
    
    def setup(self, stage):
        if not self.cache_dataset:
            self.onehot = lambda x: one_hot_labels(x, self.num_labels)
            
            self.dataset = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root='./cifar10_data', 
                train=True, 
                download=True,
                transform=self.convert_img,
                target_transform = self.onehot
                ), self.train_percentage))
           
            self.dataset['test'] = getattr(torchvision.datasets, self.task_name.upper())(root='./cifar10_data',
                train=False,
                download=True,
                transform=self.convert_img,
                target_transform = self.onehot
            )
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
        else:
            self.onehot = lambda x: one_hot_labels(x, self.num_labels)
            self.dataset = {}
            self.dataset['train'] = split_stratify(tensor_dataset(getattr(torchvision.datasets, self.dataset_names[self.task_name])(root=self.cached_dataset_filepath, 
                train=True, 
                transform=self.convert_img,
                target_transform = self.onehot
                ), self.train_percentage))
           
            self.dataset['test'] = getattr(torchvision.datasets, self.task_name.upper())(root=self.cached_dataset_filepath,
                train=False,
                transform=self.convert_img,
                target_transform = self.onehot
            )
            self.dataset['validation'] = copy.deepcopy(self.dataset['test'])
            
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]