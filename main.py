import random
from pyDOE import lhs
import math
import numpy as np
import pandas as pd
from cec17_functions import cec17_test_func
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt



def func(specimen, valueToReturn, numOfDimensions, number, functionNumber):
    if functionNumber == 0:
        valueToReturn[0] = sum(100.0 * np.power( (np.subtract(specimen[1:], np.power(specimen[:-1], 2)) ), 2) + np.power((np.subtract(1, specimen[:-1])), 2))
    else:
        valueToReturn[0] = sum([ x**2 for x in specimen ])

class DEParams:
    populationSize = 100
    crossoverProbability = 0.5
    differentialWeight = 0.8
    maxfes = 2000

    penaltyFactor = 0.1
    evaluationFunction = staticmethod(func)
    # evaluationFunction = staticmethod(cec17_test_func)

    #for surogate model
    trainingDataSize = 10000



class Common:
    @staticmethod
    def initialization(populationSize):
        return np.random.uniform(minValue, maxValue, (populationSize, dimensions))

    @staticmethod
    def generate(population, index, params):
        indexSet = { index }
        individuals = [ population[i] for i in random.sample( set(range(params.populationSize)) - indexSet, 3 ) ]
        return individuals

    @staticmethod
    def mutation(individuals, params):
        donorVector = np.add(individuals[0], params.differentialWeight * np.array(np.subtract(individuals[1], individuals[2])))
        return donorVector

    @staticmethod
    def crossover(specimen, donorVector, params):
        randomI = random.sample(range(dimensions),1)
        trialVector = [ donorVector[i] if random.uniform(0,1) <= params.crossoverProbability or i == randomI else specimen[i] for i in range(dimensions) ]
        return trialVector

    @staticmethod
    def constraintViolation(x):
        minViolation = minValue - x
        maxViolation = x - maxValue
        if minViolation > maxViolation:
            value = minViolation
        else:
            value = maxViolation

        return value

    @staticmethod
    def penalty(vector, params):
        arr = np.array([ params.penaltyFactor * max( 0, Common.constraintViolation(x) ) for x in vector ])
        penaltyValue = arr.sum()
        return penaltyValue




class DE:
    @staticmethod
    def evaluate(y, x, population, params):
        global dimensions
        global funNumCEC
        x_val = [0]
        y_val = [0]
        params.evaluationFunction(x, x_val, dimensions, 1, funNumCEC)
        params.evaluationFunction(y, y_val, dimensions, 1, funNumCEC)

        x_val += Common.penalty(x, params)
        y_val += Common.penalty(y, params)
        if y_val <= x_val:
            population[np.where( population == x )[0][0]] = y


    def run(self, params):
        global generationNum
        population = Common.initialization(params.populationSize)
        fes = 0
        generationNum = 0
        while fes < params.maxfes:
            for specimen in population:
                individuals = Common.generate(population, np.where( population == specimen )[0][0], params)
                donorVector = Common.mutation(individuals, params)
                trialVector = Common.crossover(specimen, donorVector, params)
                DE.evaluate(trialVector, specimen, population, params)
                fes+=1
            generationNum+=1
        return population





class DEANN:
    @staticmethod
    def generateTrainingData(params):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(dimensions, samples=params.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(params.trainingDataSize * 0.8)
        return data[:cut, :], data[cut:, :]

    @staticmethod
    def dataNormalization(training, validation):
        normalizerTraining = Normalization(axis=-1)
        normalizerTraining.adapt(training)
        normalizerValidation = Normalization(axis=-1)
        normalizerValidation.adapt(validation)

        return normalizerTraining(training), normalizerValidation(validation)

    @staticmethod
    def evaluateSet(set, params):
        global evaluated
        evaluated = []
        for x in set:
            val = [0]
            params.evaluationFunction(x, val, dimensions, 1, funNumCEC)
            evaluated.append(val)
        return np.array(evaluated)

    @staticmethod
    def evaluateWithModel(y, x, population, model, params):
        x_val = model.predict(np.array(x).reshape(1,dimensions))

        y_val = model.predict(np.array(y).reshape(1,dimensions))

        x_val += Common.penalty(x, params)
        y_val += Common.penalty(y, params)

        if y_val <= x_val:
            population[np.where( population == x )[0][0]] = y

    def getModel(self, params):
        training, validation = DEANN.generateTrainingData(params)

        evaluatedTraining = DEANN.evaluateSet(training, params)
        evaluatedValidation = DEANN.evaluateSet(validation, params)

        model = keras.Sequential()
        model.add(keras.Input(shape=(dimensions)))

        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(20, activation=keras.activations.relu, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))

        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(20, activation=keras.activations.relu, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))

        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(20, activation=keras.activations.relu, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))

        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(1, activation=keras.activations.linear, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))

        model.compile(optimizer=keras.optimizers.Adam(clipnorm=1), loss=keras.losses.MeanSquaredError())

        model.fit(training, evaluatedTraining, epochs=10, validation_data=(validation, evaluatedValidation))
        return model

    def run(self, params, model):
        global generationNum
        population = Common.initialization(params.populationSize)





        fes = 0
        generationNum = 0
        while fes < params.maxfes:
            for specimen in population:
                individuals = Common.generate(population, np.where( population == specimen )[0][0], params)
                donorVector = Common.mutation(individuals, params)
                trialVector = Common.crossover(specimen, donorVector, params)
                DEANN.evaluateWithModel(trialVector, specimen, population, model, params)
                fes += 1
                #print_hi(fes)
            generationNum += 1
            # evaluatedPopulation = DEANN.evaluateSet(population, params)
            # model.fit(population, evaluatedPopulation, epochs=10, validation_data=(validation, evaluatedValidation))
        return population

if __name__ == '__main__':

    global minValue
    global maxValue
    global dimensions
    global generationNum
    global funNumCEC
    minValue = -100
    maxValue = 100
    generationNum = 0
    dimensions = 5  #only: 2, 10, 20, 30, 50, 100
    funNumCEC = 1

    params = DEParams()
    iters = 10

    DE_alg = DE()
    for i in range(iters):
        print("##### "+ str(i+1)+ " #####")
        population = DE_alg.run(params)
        best = population[0]
        best_val = [0]
        params.evaluationFunction(best, best_val, dimensions, 1, funNumCEC)
        for s in population:
            s_val = [0]
            params.evaluationFunction(s, s_val, dimensions, 1, funNumCEC)
            if s_val <= best_val:
                best = s
                best_val = s_val
        #print("Najlepszy: " +str(best) )
        #print("Wartość: ")
        print(best_val)
        #print("Numer generacji:" + str(generationNum))


    DEANN_alg = DEANN()
    model = DEANN_alg.getModel(params)

    for i in range(iters):
        print("##### "+ str(i+1)+ " #####")
        population = DEANN_alg.run(params, model)
        best = population[0]
        best_val = [0]
        params.evaluationFunction(best, best_val, dimensions, 1, funNumCEC)
        for s in population:
            s_val = [0]
            params.evaluationFunction(s, s_val, dimensions, 1, funNumCEC)
            if s_val <= best_val:
                best = s
                best_val = s_val
        #print("Najlepszy: " + str(best) )
        #print("Wartość: ")
        print(best_val)
        #print("Numer generacji:" + str(generationNum))


