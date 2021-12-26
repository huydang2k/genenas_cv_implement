import argparse
import pytorch_lightning as pl

from problem import CV_DataModule
from problem import CV_Problem_MultiObjNoTrain, NasgepNet
from evolution import MultiObjectiveOptimizer

import logging

logging.disable(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = CV_Problem_MultiObjNoTrain.add_arguments(parser)
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CV_DataModule.add_argparse_args(parser)
    parser = CV_DataModule.add_cache_arguments(parser)
    parser = NasgepNet.add_model_specific_args(parser)
    parser = NasgepNet.add_learning_specific_args(parser)
    parser = MultiObjectiveOptimizer.add_optimizer_specific_args(parser)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    # print(args.input_size)
    # print('adsdsa')
    # print(args)
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
    # get args
    args = parse_args()
    print(args)
    # solve source problems
    problem = CV_Problem_MultiObjNoTrain(args)

    # create optimizer
    optimizer = MultiObjectiveOptimizer(args)
    # optimizer.add_optimizer_specific_args(args)
    # Optimize architecture
    print('Run')
    population, objs = optimizer.ga(problem)

    for i, idv in enumerate(population):
        symbols, _, _ = problem.replace_value_with_symbol(population[i])
        print(f"Individual {i + 1}: {objs[i]}, chromosome: {symbols}")
        problem.make_graph(idv, prefix=f"{args.task_name}.idv_{i+1}")

    # build and save model
    # lb, ub = problem.get_bounds()
    # model = amt.MultinomialModel(population, lb, ub)
    # amt.util.save_model(model, args.task_name)


if __name__ == "__main__":
    main()
