from datetime import datetime
import os
import argparse
import pytorch_lightning as pl
from problem import CV_DataModule_train, CV_Problem_MultiObjTrain, NasgepNet_multiObj
import numpy as np
import pickle
from problem import nasgep_net_train
path = os.path.dirname(os.path.abspath(__file__))
today = datetime.today().strftime("%Y-%m-%d")

# run_loss = {}
# run_accuracy = {}
# chromosome_index = -1
def input_chromosome(args):
    try:
        with open(args.checkpoint_population_file,'rb') as f:
            d = pickle.load(f)
            fitness = [i[2] for i in d['fitness']]
            # best_indicies = np.argsort(fitness)
            return np.array(d['population'])
            
    except:
        print('Read from txt fle')
    try:
        with open(args.file_name, 'rb') as f:
            chromosome = pickle.load(f)        
    except:
        with open(path + args.file_name, 'r') as f:
            chromosome = f.read()
    try:
        chromosome = chromosome.split()
        chromosome = [int(x) for x in chromosome]
        return np.array(chromosome)
        
    except:
        return np.array(chromosome)


    


def parse_args():
    parser = argparse.ArgumentParser()
    parser = CV_Problem_MultiObjTrain.add_train_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CV_DataModule_train.add_argparse_args(parser)
    parser = CV_DataModule_train.add_cache_arguments(parser)
    parser.add_argument("--file_name", default= '/chromosome.txt', type=str)
    parser.add_argument("--checkpoint_population_file", default= '/checkpoint.pkl', type=str)
    parser.add_argument("--save_path", default = path + f"chromosome_trained_weights.gene_nas.{today}.pkl", type= str)
    parser = CV_Problem_MultiObjTrain.add_arguments(parser)
    parser = NasgepNet_multiObj.add_model_specific_args(parser)
    parser = NasgepNet_multiObj.add_learning_specific_args(parser)
    args = parser.parse_args()

    args.num_terminal = args.num_main + 1
    args.l_main = args.h_main * (args.max_arity - 1) + 1
    args.l_adf = args.h_adf * (args.max_arity - 1) + 1
    args.main_length = args.h_main + args.l_main
    args.adf_length = args.h_adf + args.l_adf
    args.chromosome_length = (
        args.num_main * args.main_length + args.num_adf * args.adf_length
    )
    args.D = args.chromosome_length
    args.mutation_rate = args.adf_length / args.chromosome_length

    return args


        
def main():
    
    args = parse_args()
    # print(args)
    problem = CV_Problem_MultiObjTrain(args= args)
    chromosome_ls = input_chromosome(args)
    #get the best chromosome
    chromosome_ls = [chromosome_ls[19]]
    for chromosome in chromosome_ls:
        nasgep_net_train.chromosome_index += 1
        nasgep_net_train.run_loss[str(nasgep_net_train.chromosome_index)] = []
        nasgep_net_train.run_accuracy[str(nasgep_net_train.chromosome_index)] = []
        print(nasgep_net_train.chromosome_index)
        problem.evaluate(chromosome= chromosome)
    log_path = args.save_path.replace('.pkl','.log_infor.pkl')
    saved = {'acc': nasgep_net_train.run_accuracy, 'loss': nasgep_net_train.run_loss}
    with open(f'logs/{log_path}','wb') as f:
        pickle.dump(saved,f)
    

if __name__ == "__main__":
    main()