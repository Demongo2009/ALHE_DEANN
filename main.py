# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import math
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def func(y):
    return sum([ x**2 for x in y ])

class DEParams:
    populationSize = 100
    mutationFactor = 0.1
    crossoverProbability = 0.9
    differentialWeight = 0.8
    maxfes = 20000
    evaluationFunction = staticmethod(func)


def initialization(populationSize):
    population = []
    for i in range(populationSize):
        population.append([ random.uniform(minValue, maxValue) for j in range(dimensions) ])
    return population


def generate(population, index, params):
    indexSet = { index }
    individuals = [ population[i] for i in random.sample( set(range(params.populationSize)) - indexSet, 3 ) ]
    return individuals


def mutation(individuals, params):
    donorVector = np.add(individuals[0], params.differentialWeight * np.array(np.subtract(individuals[1], individuals[2])))
    return donorVector


def crossover(specimen, donorVector, params):
    randomI = random.sample(range(dimensions),1)
    trialVector = [ donorVector[i] if random.uniform(0,1) <= params.crossoverProbability or i == randomI else specimen[i] for i in range(dimensions) ]
    return trialVector


def evaluate(y, x, population, params):
    if params.evaluationFunction(y) <= params.evaluationFunction(x):
        population[population.index(x)] = y






def DE(params):
    global generationNum
    population = initialization(params.populationSize)
    fes = 0
    while fes < params.maxfes:
        for specimen in population:
            individuals = generate(population, population.index(specimen), params)
            donorVector = mutation(individuals, params)
            trialVector = crossover(specimen, donorVector, params)
            evaluate(trialVector, specimen, population, params)
            fes+=1
        generationNum+=1
    return population



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    global minValue
    global maxValue
    global dimensions
    global generationNum
    minValue = -100
    maxValue = 100
    generationNum = 0
    dimensions = 6
    dEParams = DEParams()
    population = DE(dEParams)
    print(population)
    best = population[0]
    for s in population:
        if dEParams.evaluationFunction(s) <= dEParams.evaluationFunction(best):
            best = s
    print(best)
    print(dEParams.evaluationFunction(best))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
