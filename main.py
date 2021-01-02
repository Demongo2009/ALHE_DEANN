# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import math
import numpy as np
from cec17_functions import cec17_test_func
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers

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
    maxfes = 200000
    evaluationFunction = staticmethod(cec17_test_func)


def initialization(populationSize):
    # population = []
    # for i in range(populationSize):
    #     population.append([ random.uniform(minValue, maxValue) for j in range(dimensions) ])
    # return np.array(population)
    return np.random.uniform(minValue, maxValue, (populationSize, dimensions))


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
        population[np.where( population == x )[0][0]] = y





def DE(params):
    global generationNum
    population = initialization(params.populationSize)
    fes = 0
    while fes < params.maxfes:
        for specimen in population:
            individuals = generate(population, np.where( population == specimen )[0][0], params)
            donorVector = mutation(individuals, params)
            trialVector = crossover(specimen, donorVector, params)
            evaluate(trialVector, specimen, population, params)
            fes+=1
        generationNum+=1
    return population






class DEANNParams:
    populationSize = 100
    mutationFactor = 0.1
    crossoverProbability = 0.9
    differentialWeight = 0.8
    maxfes = 20000
    trainingDataSize = 10000
    evaluationFunction = staticmethod(func)

def generateTrainingData(params):
    data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
    return data[:80, :], data[80:, :]


def dataNormalization(training, validation):
    normalizerTraining = Normalization(axis=-1)
    normalizerTraining.adapt(training)
    normalizerValidation = Normalization(axis=-1)
    normalizerValidation.adapt(validation)

    return normalizerTraining(training), normalizerValidation(validation)


def evaluateSet(set, params):
    evaluated = np.array([ np.float32(params.evaluationFunction(x)) for x in set ])
    return evaluated

def evaluateWithModel(y, x, population, model):
    if model.predict(np.array(y).reshape(1,6)) <= model.predict(np.array(x).reshape(1,6)):
        population[np.where( population == x )[0][0]] = y



def DEANN(params):
    global generationNum
    population = initialization(params.populationSize)


    training, validation = generateTrainingData(params)
    normTraining, normValidation = dataNormalization(training, validation)
    evaluatedTraining = evaluateSet(training, params)
    evaluatedValidation = evaluateSet(validation, params)

    inputs = keras.Input(shape=(dimensions))

    x = layers.Dense(50, activation='relu')(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(50, activation='relu')(x)

    outputs = layers.Dense(1, activation='relu')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanAbsoluteError())

    model.fit(normTraining, evaluatedTraining, batch_size=np.int32(params.trainingDataSize * 0.8), epochs=1000, validation_data=(normValidation,evaluatedValidation))


    fes = 0
    while fes < params.maxfes:
        for specimen in population:
            individuals = generate(population, np.where( population == specimen)[0][0], params)
            donorVector = mutation(individuals, params)
            trialVector = crossover(specimen, donorVector, params)
            evaluateWithModel(trialVector, specimen, population, model)
            fes += 1
        generationNum += 1
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
    print(population)
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



    dEANNParams = DEANNParams()
    population = DEANN(dEANNParams)
    print(population)
    best = population[0]
    for s in population:
        if dEANNParams.evaluationFunction(s) <= dEANNParams.evaluationFunction(best):
            best = s
    print(best)
    print(dEANNParams.evaluationFunction(best))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
