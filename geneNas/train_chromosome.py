import os
import argparse
import pytorch_lightning as pl
from problem import CV_DataModule_train, CV_Problem_MultiObjTrain, NasgepNet_multiObj
import numpy as np
import pickle

path = os.path.dirname(os.path.abspath(__file__))

def input_chromosome(file: str):
    try:
        with open(file, 'rb') as f:
            chromosome = pickle.load(f)        
    except:
        with open(path + file, 'r') as f:
            chromosome = f.read()
    try:
        chromosome = chromosome.split()
        chromosome = [int(x) for x in chromosome]
        return np.array(chromosome)
        
    except:
        return np.array(chromosome)


    


def parse_args():
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = CV_DataModule_train.add_argparse_args(parser)
    parser = CV_DataModule_train.add_cache_arguments(parser)
    parser.add_argument("--file_name", default= '/chromosome', type=str)
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
    chromosome = input_chromosome(args.file_name)
    problem = CV_Problem_MultiObjTrain(args= args)
    problem.evaluate(chromosome= chromosome)
    
    

if __name__ == "__main__":
    main()