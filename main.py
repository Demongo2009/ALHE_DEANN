# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import math
import numpy as np
from cec17_functions import cec17_test_func

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def func(y):
    return sum([ x**2 for x in y ])

class DEParams:
    populationSize = 1000
    mutationFactor = 0.1
    crossoverProbability = 0.9
    differentialWeight = 0.8
    maxfes = 200000
    evaluationFunction = staticmethod(cec17_test_func)


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
    global dimensions
    global funNumCEC
    x_val = [0]
    y_val = [0]
    params.evaluationFunction(x, x_val, dimensions, 1, funNumCEC)
    params.evaluationFunction(y, y_val, dimensions, 1, funNumCEC)
    if y_val <= x_val:
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
    global funNumCEC
    minValue = -100  #cannot find what are the limits for this function
    maxValue = 100
    generationNum = 0
    dimensions = 10  #only: 2, 10, 20, 30, 50, 100
    funNumCEC = 1

    dEParams = DEParams()
    population = DE(dEParams)
    best = population[0]
    best_val = [0]
    dEParams.evaluationFunction(best, best_val, dimensions, 1, funNumCEC)
    for s in population:
        s_val = [0]
        dEParams.evaluationFunction(s, s_val, dimensions, 1, funNumCEC)
        if s_val <= best_val:
            best = s
            best_val = s_val
    print("Najlepszy: ")
    print( str(best))
    print("Wartość: ")
    print(best_val)

    print("Numer generacji:" + str(generationNum))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
