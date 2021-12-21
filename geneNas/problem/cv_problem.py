#Implement search strategy
from .nasgep_net import NasgepNet
from .abstract_problem import Problem
from .function_set import CV_Main_FunctionSet, CV_ADF_FunctionSet
from typing import List, Tuple
import numpy as np
from .cv_data_module import CV_DataModule
from util.logger import ChromosomeLogger
from evolution import GeneType
import time
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.nn import functional as F
from tqdm import tqdm
class CV_Problem_MultiObjNoTrain(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.main_function_set = CV_Main_FunctionSet.return_func_name()
        self.adf_function_set = CV_ADF_FunctionSet.return_func_name()
        self.dm = CV_DataModule.from_argparse_args(self.hparams)
        self.dm.prepare_data()
        self.dm.setup("fit")

        self.chromsome_logger = ChromosomeLogger()
        self.metric_name = self.dm.metrics_names[self.hparams.task_name]
        
        self.progress_bar = 0
        self.weights_summary = None
        self.early_stop = None
        self.k_folds = self.hparams.k_folds
        
        self.weight_values = [0.5, 1, 2, 3]
        self.metric = self.dm.metric
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
        # self.replace_value_with_symbol(individual)
        # print('parse chromosome')
        # print('INFOR: num main {}, main length {}, adf length {}, num adf {}'.format(
        #     self.hparams.num_main,
        #     self.hparams.main_length,
        #     self.hparams.num_adf,
        #     self.hparams.adf_length
        # ))
        
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
            
            adf_func[f"a{i + 1}"] = adf

        for i in range(self.hparams.num_main):
            start_idx = i * self.hparams.main_length
            end_idx = start_idx + self.hparams.main_length
            # print('Decode main ',i)
            # print(chromosome[start_idx:end_idx])
            sub_chromosome = chromosome[start_idx:end_idx]
            main_func = self.parse_tree(sub_chromosome, main_function_set)
            
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
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders
        )
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print(f"New optimal lr: {new_lr}")
    def apply_weight(self, model, value):
        sampler = torch.distributions.uniform.Uniform(low=-value, high=value)
        with torch.no_grad():
            for name, param in model.named_parameters():
                new_param = sampler.sample(param.shape)
                param.copy_(new_param)
        return
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
        )
        return trainer

    def setup_model(self, chromosome):
        self.chromsome_logger.log_chromosome(chromosome)
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)
        
        glue_pl = NasgepNet(
            num_labels=self.dm.num_labels,
            eval_splits=self.dm.eval_splits,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        glue_pl.init_model(mains, adfs)
        glue_pl.init_chromosome_logger(self.chromsome_logger)
        return glue_pl
    
    def run_inference(self, model, weight_value, val_dataloader):
        self.apply_weight(model, weight_value)
        model.cuda()
        outputs = []
        encounter_nan = False
        for batch in tqdm(val_dataloader):
            labels = batch[1]
            batch =  batch[0]['feature_map'].cuda()
            logits = model(batch)

            if self.dm.num_labels > 1:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = logits.squeeze()
            preds = preds.detach().cpu()
            # batch = {k: v.detach().cpu() for k, v in batch.items()}

            outputs.append({"preds": preds, "labels": labels})


        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        

        metrics = self.metric.compute(predictions=preds, references=labels)[
            self.metric_name
        ]
        
        return metrics


    def perform_kfold(self, model):
        avg_metrics = 0
        avg_max_metrics = 0
        total_time = 0

        for fold, _, val_dataloader in self.dm.kfold(self.k_folds, None):
            start = time.time()
            metrics = [
                self.run_inference(model, wval, val_dataloader)
                for wval in self.weight_values
            ]
            end = time.time()
            avg_metrics += np.mean(metrics)
            avg_max_metrics += np.max(metrics)
            total_time += end - start
            print(
                f"FOLD {fold}: {self.metric_name} {np.mean(metrics)} {np.max(metrics)} ; Time {end - start}"
            )

        # result = trainer.test()
        avg_metrics = avg_metrics / self.k_folds
        avg_max_metrics = avg_max_metrics / self.k_folds
        print(
            f"FOLD AVG: {self.metric_name} {avg_metrics} {avg_max_metrics} ; Time {total_time}"
        )
        return avg_metrics, avg_max_metrics

    def evaluate(self, chromosome: np.array):
        #fix here
       
        print('Evaluate primitive chrosome')
        print(chromosome)
        
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        nasgep_model = self.setup_model(chromosome)
        return self.perform_kfold(nasgep_model)
