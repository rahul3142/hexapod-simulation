import numpy as np
import random
from pyeasyga import pyeasyga
from Body_5 import evolve_gaits


data = np.arange(0,42,1)

gen_alg = pyeasyga.GeneticAlgorithm(data,
                               population_size=15,
                               generations=20,
                               crossover_probability=1,
                               mutation_probability=0.2,
                               elitism=True,
                               maximise_fitness=True)

def create_individual(data):
    individual = []
    for i in data:
        p = random.random()
        individual.append(p)
        
    return individual

gen_alg.create_individual = create_individual


#a = np.random.rand(1,len(data))

def fitness_function(individual,data):
    weights = np.reshape(individual,[7,6])
    a,b,c = evolve_gaits(weights)
    fitness = -2*a-10*b-10*c
    return fitness

gen_alg.fitness_function = fitness_function

gen_alg.run()

a = gen_alg.best_individual()

print(gen_alg.best_individual())
