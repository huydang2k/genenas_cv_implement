import pickle
import numpy as np
import torch
# PATH = '/hdd/huydang/genenas_cv_implement/geneNaschromosome_trained_weights.gene_nas.2021-12-28.pkl'
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# print(model)
with open('cifar10.gene_nas.2021-12-28_gauss.pkl','rb') as f:
    d = pickle.load(f)
    fitness = [i[2] for i in d['fitness']]
    best_indicies = np.argsort(fitness)
    print(d['population'][best_indicies[-1]])
    print(d)


# def uniform_crossover( population):
#         print("population")
#         print(population)
#         # extract parameters
#         N, D = population.shape

#         # select for crossover
#         parent1 = population[np.random.permutation(N), :]
#         parent2 = population[np.random.permutation(N), :]
#         print(parent1)
#         print(parent2)
#         offspring = np.zeros([N, D])

#         # create random variable
#         r = np.random.rand(N, D)

#         # uniform crossover
#         index = r >= 0.5
#         offspring[index] = parent1[index]
#         index = r < 0.5
#         offspring[index] = parent2[index]
#         print(offspring)
#         return offspring.astype(np.int32)


# uniform_crossover(d['population'][:4])