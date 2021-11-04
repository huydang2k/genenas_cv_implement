#Implement search strategy
from nasgep_net import NasgepNet
from abstract_problem import Problem
from function_set import CV_Main_FunctionSet, CV_ADF_FunctionSet

class CV_Problem_MultiObjNoTrain(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.function_set = C.return_func_name()
        self.dm = DataModule.from_argparse_args(self.hparams)
        self.dm.prepare_data()
        self.dm.setup("fit")

        self.chromsome_logger = ChromosomeLogger()
        self.metric_name = self.dm.metrics_names[self.hparams.task_name]

        self.progress_bar = 0
        self.weights_summary = None
        self.early_stop = None

        self.k_folds = self.hparams.k_folds
    
    def parse_chromosome(
        self, chromosome: np.array, main_function_set=CV_Main_FunctionSet,adf_function_set = CV_ADF_FunctionSet, return_adf=False
    ):
        # self.replace_value_with_symbol(individual)
        # print('parse chromosome')
        print('INFOR: num main {}, main length {}, adf length {}, num adf {}'.format(
            self.hparams.num_main,
            self.hparams.main_length,
            self.hparams.num_adf,
            self.hparams.adf_length
        ))
        
        total_main_length = self.hparams.num_main * self.hparams.main_length
        # print('total main length ',total_main_length)
        all_main_func = []
        adf_func = {}
        #Split into sublist, each present a main or an adf
        for i in range(self.hparams.num_adf):
            start_idx = total_main_length + i * self.hparams.adf_length
            end_idx = start_idx + self.hparams.adf_length
            print('Decode adf ',i)
            print(chromosome[start_idx:end_idx])
            sub_chromosome = chromosome[start_idx:end_idx]
            adf = self.parse_tree(sub_chromosome, adf_function_set)
            print(type(adf))
            adf_func[f"a{i + 1}"] = adf

        for i in range(self.hparams.num_main):
            start_idx = i * self.hparams.main_length
            end_idx = start_idx + self.hparams.main_length
            print('Decode main ',i)
            print(chromosome[start_idx:end_idx])
            sub_chromosome = chromosome[start_idx:end_idx]
            main_func = self.parse_tree(sub_chromosome, main_function_set)
            print(type(main_func))
            # main_func.assign_adfs(main_func.root, adf_func)
            all_main_func.append(main_func)

        if return_adf:
            return all_main_func, adf_func
        else:
            return all_main_func

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
        print('mains: ', mains, type(mains[0]))
        print('adfs:  ', adfs,  type(adfs))
        glue_pl = NasgepNet(
            num_labels=self.dm.num_labels,
            eval_splits=self.dm.eval_splits,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        glue_pl.init_model(mains, adfs)
        glue_pl.init_chromosome_logger(self.chromsome_logger)
        return glue_pl

    def perform_kfold(self, model):
        avg_metrics = 0
        total_time = 0
        print('KFOLD  ',self.k_folds)
        trainer = self.setup_trainer()
        print('SET up trainer-------')
        print(type(trainer))
        model.reset_weights()
        _, train_dataloader, val_dataloader = next(self.dm.kfold(self.k_folds, None))
        self.lr_finder(model, trainer, train_dataloader, val_dataloader)

        for fold, train_dataloader, val_dataloader in self.dm.kfold(self.k_folds, None):
            start = time.time()
            try:
                model.reset_weights()
                trainer = self.setup_trainer()
                print('Set up trainer--')
                print(type(trainer))
                trainer.fit(
                    model,
                    train_dataloader=train_dataloader,
                    val_dataloaders=val_dataloader,
                )
                metrics = self.chromsome_logger.logs[-1]["data"][-1]["metrics"][
                    self.metric_name
                ]
            except NanException as e:
                print(e)
                log_data = {
                    f"val_loss": 0.0,
                    "metrics": {self.metric_name: 0.0},
                    "epoch": -1,
                }
                metrics = log_data["metrics"][self.metric_name]
            end = time.time()
            avg_metrics += metrics
            total_time += end - start
            print(f"FOLD {fold}: {self.metric_name} {metrics} ; Time {end - start}")

        # result = trainer.test()
        avg_metrics = avg_metrics / self.k_folds
        print(f"FOLD AVG: {self.metric_name} {avg_metrics} ; Time {total_time}")
        return avg_metrics

    def evaluate(self, chromosome: np.array):
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        glue_pl = self.setup_model(chromosome)
        return self.perform_kfold(glue_pl)
