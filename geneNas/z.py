
import numpy as np

def uniform_crossover( population):
        print(population)
        # extract parameters
        N, D = population.shape

        # select for crossover
        parent1 = population[np.random.permutation(N), :]
        parent2 = population[np.random.permutation(N), :]
        print('prr')
        print(parent1, type(parent1))
        print(parent2, type(parent2))
        offspring = np.zeros([N, D])

        # create random variable
        r = np.random.rand(N, D)
        print('index')
        # uniform crossover
        index = r >= 0.5
        print(index, type(index))
        offspring[index] = parent1[index]
        print(offspring[index] , type(offspring[index]))
        index = r < 0.5
        offspring[index] = parent2[index]
        print(index, type(index))
        print(offspring[index], type(offspring[index]))
        print(offspring, type(offspring))
        return offspring.astype(np.int32)

def mutate(offspring):
        print(offspring, type(offspring))
        # extract parameters
        N, D = offspring.shape
        print('r')
        # create random variable
        r = np.random.rand(N, D)
        print(r, type(r))
        # mutate with p=1/D
        print('index')
        index = r < 1.0 / float(D)
        print(index, type(index))
        offspring[index] = np.random.randint(
            low=[1,2,3], high=[7,8,9], size=offspring.shape
        )[index]
        print(offspring[index], type(offspring[index]))
        print(offspring,type(offspring[index]))
        return offspring.astype(np.int32)

s = np.random.rand(4,3)

mutate(s)