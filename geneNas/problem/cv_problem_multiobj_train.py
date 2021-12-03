#Implement search strategy
from .nasgep_net_train import NasgepNetRWE_multiObj, NasgepNet_multiObj
from .abstract_problem import Problem
from .function_set import CV_Main_FunctionSet, CV_ADF_FunctionSet
from typing import List, Tuple
import numpy as np
from .cv_data_module import CV_DataModule_train
from util.logger import ChromosomeLogger
from evolution import GeneType
import time
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from util.exception import NanException
from torch.utils.data import DataLoader


class CV_Problem_MultiObjTrain(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.main_function_set = CV_Main_FunctionSet.return_func_name()
        self.adf_function_set = CV_ADF_FunctionSet.return_func_name()
        self.dm = CV_DataModule_train.from_argparse_args(self.hparams)
        self.dm.prepare_data()
        self.dm.setup("fit")

        self.metric_name = self.dm.metrics_names[self.hparams.task_name]
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.eval_batch_size
        self.chromsome_logger = ChromosomeLogger()
        self.save_path = args.save_path
        self.progress_bar = 1
        self.weights_summary = None
        self.early_stop = None
        self.k_folds = self.hparams.k_folds
        # self.weight_values = [0.5, 1, 2, 3]
        
        
    def _get_chromosome_range(self) -> Tuple[int, int, int, int,int]:
        R1 = len(self.main_function_set)
        R2 = R1 + self.hparams.num_adf
        R3 = R2 + self.hparams.num_terminal
        R4 = R3 + len(self.adf_function_set)
        R5 = R4 + self.hparams.max_arity
        
        return R1, R2, R3, R4, R5 
    def get_feasible_range(self, idx) -> Tuple[int, int]:
        # Generate lower_bound, upper_bound for gene at given index of chromosome
        R1, R2, R3, R4, R5 = self._get_chromosome_range()
        # gene at index idx belong to one of the given mains
        total_main_length = self.hparams.num_main * self.hparams.main_length
        if idx < total_main_length:
            if idx % self.hparams.main_length < self.hparams.h_main:
                # Head of main: adf_set and function_set
                return 0, R2
            else:
                # Tail of main: terminal_set
                return R2, R3
        if (idx - total_main_length) % self.hparams.adf_length < self.hparams.h_adf:
            # Head of ADF: function_set
            return R3, R4
        else:
            # Tail of ADF: adf_terminal_set
            return R4, R5

    def parse_chromosome(
        self, chromosome: np.array, main_function_set=CV_Main_FunctionSet,adf_function_set = CV_ADF_FunctionSet, return_adf=False
    ):

        total_main_length = self.hparams.num_main * self.hparams.main_length
        # print('total main length ',total_main_length)
        all_main_func = []
        adf_func = {}
        #Split into sublist, each present a main or an adf
        for i in range(self.hparams.num_adf):
            start_idx = total_main_length + i * self.hparams.adf_length
            end_idx = start_idx + self.hparams.adf_length
            # print('Decode adf ',i)
            # print(chromosome[start_idx:end_idx])
            sub_chromosome = chromosome[start_idx:end_idx]
            adf = self.parse_tree(sub_chromosome, adf_function_set)
            
            # print(type(adf))
            adf_func[f"a{i + 1}"] = adf

        for i in range(self.hparams.num_main):
            start_idx = i * self.hparams.main_length
            end_idx = start_idx + self.hparams.main_length
            # print('Decode main ',i)
            # print(chromosome[start_idx:end_idx])
            sub_chromosome = chromosome[start_idx:end_idx]
            main_func = self.parse_tree(sub_chromosome, main_function_set)
            # print(type(main_func))
            # main_func.assign_adfs(main_func.root, adf_func)
            all_main_func.append(main_func)

        if return_adf:
            return all_main_func, adf_func
        else:
            return all_main_func

    def replace_value_with_symbol(
        self, chromosome: np.array
    ) -> Tuple[List, List, List]:
        # create GEP symbols from integer chromosome
        symbols = []
        arity = []
        gene_types = []
        R1, R2, R3, R4, R5 = self._get_chromosome_range()
        for i, value in enumerate(chromosome):
            value = int(value)
            if value >= R4: #adf input
                symbols.append(self.adf_terminal_name[value - R4])
                arity.append(0)
                gene_types.append(GeneType.ADF_TERMINAL)
            elif value >= R3: #adf function
                symbols.append(self.adf_function_set[value-R3]["name"])
                arity.append(self.adf_function_set[value-R3]["arity"])
                gene_types.append(GeneType.FUNCTION)
            elif value >= R2: #main variables
                symbols.append(self.terminal_name[value - R2])
                arity.append(0)
                gene_types.append(GeneType.TERMINAL)
            elif value >= R1: #adf name
                symbols.append(self.adf_name[value - R1])
                arity.append(self.hparams.max_arity)
                gene_types.append(GeneType.ADF)
            else: #main functions
                symbols.append(self.main_function_set[value]["name"])
                arity.append(self.main_function_set[value]["arity"])
                gene_types.append(GeneType.FUNCTION)
        return symbols, arity, gene_types

    @staticmethod
    def total_params(model):
        return sum(p.numel() for p in model.parameters())

    def lr_finder(self, model, trainer, train_dataloader, val_dataloaders):
        lr_finder = trainer.tuner.lr_find(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders
        )
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print(f"New optimal lr: {new_lr}")
        
    # def apply_weight(self, model, value):
    #     sampler = torch.distributions.uniform.Uniform(low=-value, high=value)
    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             new_param = sampler.sample(param.shape)
    #             param.copy_(new_param)
    #     return
    
    def setup_model_trainer(self, chromosome: np.array):
        glue_pl = self.setup_model(chromosome)
        trainer = self.setup_trainer()
        return glue_pl, trainer

    def setup_trainer(self):
        if type(self.early_stop) == int:
            early_stop = EarlyStopping(
                monitor=self.metric_name,
                min_delta=0.00,
                patience=self.early_stop,
                verbose=False,
                mode="max",
                
            )
            early_stop = [early_stop]
        else:
            early_stop = None

        trainer = pl.Trainer.from_argparse_args(
            self.hparams,
            progress_bar_refresh_rate=self.progress_bar,
            # automatic_optimization=False,
            weights_summary=self.weights_summary,
            checkpoint_callback=False,
            callbacks=early_stop,
            max_epochs = self.hparams.max_epochs,
            check_val_every_n_epoch=1,
        )
        return trainer

    def setup_model(self, chromosome):
        
        self.chromsome_logger.log_chromosome(chromosome)
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)
        glue_pl = NasgepNet_multiObj(
            num_labels=self.dm.num_labels,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        glue_pl.init_model(mains, adfs)
        glue_pl.init_chromosome_logger(self.chromsome_logger)
        return glue_pl
    
    

    def train(self, model):
        trainer = self.setup_trainer()
        train_dataloader = DataLoader(self.dm.dataset['train'], batch_size= self.train_batch_size, shuffle= True)
        val_dataloader = DataLoader(self.dm.dataset['validation'], batch_size= self.val_batch_size)
        self.lr_finder(model, trainer, train_dataloader, val_dataloader)
        trainer.fit(
            model, 
            train_dataloaders= train_dataloader,
            val_dataloaders= val_dataloader,
        )
        # metrics = self.chromsome_logger.logs[-1]["data"][-1]["metrics"][
        #             self.metric_name
        # ]
        

    def evaluate(self, chromosome: np.array):
        print(chromosome)
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        print('Set up model')
        glue_pl = self.setup_model(chromosome)
        self.train(glue_pl)
        
        self.trainer.save_checkpoint(self.save_path)

    