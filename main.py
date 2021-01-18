import random
from pyDOE import lhs
import math
import numpy as np
import pandas as pd
from cec17_functions import cec17_test_func
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from optparse import OptionParser


class DEParams:
    def __init__(self, populationSize=100, crossoverProbability=0.7, differentialWeight=0.8, maxfes=2000,
                 dimensions=5, minValue=-100, maxValue=100, funNumCEC=1, penaltyFactor=0.1, seed=42):
        self.populationSize = populationSize
        self.crossoverProbability = crossoverProbability
        self.differentialWeight = differentialWeight
        self.maxfes = maxfes

        self.dimensions = dimensions

        self.minValue = minValue
        self.maxValue = maxValue
        self.funNumCEC = funNumCEC
        self.penaltyFactor = penaltyFactor
        self.evaluationFunction = cec17_test_func
        self.seed = seed


class DEANNParams:
    def __init__(self, trainingDataSize=10000, epochs=10, teachModelEveryGeneration=False,
                 teachModelEveryGenerationEpochs=10):
        self.trainingDataSize = trainingDataSize
        self.epochs = epochs

        self.teachModelEveryGeneration = teachModelEveryGeneration
        self.teachModelEveryGenerationEpochs = teachModelEveryGenerationEpochs


class DE:
    def __init__(self, dEParams=DEParams()):
        self.dEParams = dEParams
        self.columns = [ "x"+str(x) for x in range(dEParams.dimensions) ]
        self.columns.append("y")
        self.log = pd.DataFrame(columns=self.columns)
        self.fig = plt.figure()
        if dEParams.dimensions == 2:
            self.ax = self.fig.add_subplot(projection='3d')
        else:
            self.ax = self.fig.add_subplot()


    def initialization(self):
        return np.random.uniform(self.dEParams.minValue, self.dEParams.maxValue,
                                 (self.dEParams.populationSize, self.dEParams.dimensions))

    def generate(self, population, index):
        indexSet = {index}
        individuals = [population[i]
                       for i in random.sample(set(range(self.dEParams.populationSize)) - indexSet, 3)]
        return individuals

    def mutation(self, individuals):
        donorVector = np.add(individuals[0],
                             self.dEParams.differentialWeight * np.array(
                                 np.subtract(individuals[1], individuals[2])))
        return donorVector

    def crossover(self, specimen, donorVector):
        randomI = random.sample(range(self.dEParams.dimensions), 1)
        trialVector = [
            donorVector[i] if random.uniform(0, 1) <= self.dEParams.crossoverProbability or i == randomI else specimen[
                i]
            for i in range(self.dEParams.dimensions)]
        return trialVector

    def constraintViolation(self, specimenValue):
        minViolation = self.dEParams.minValue - specimenValue
        maxViolation = specimenValue - self.dEParams.maxValue
        if minViolation > maxViolation:
            value = minViolation
        else:
            value = maxViolation

        return value

    def penalty(self, specimen):
        arr = np.array([self.dEParams.penaltyFactor * max(0, self.constraintViolation(specimenValue))
                        for specimenValue in specimen])
        penaltyValue = arr.sum()
        return penaltyValue

    def evaluate(self, trialVector, specimen, population):
        specimen_val = [0]
        trialVector_val = [0]
        self.dEParams.evaluationFunction(specimen, specimen_val, self.dEParams.dimensions, 1, self.dEParams.funNumCEC)
        self.dEParams.evaluationFunction(trialVector, trialVector_val, self.dEParams.dimensions, 1,
                                         self.dEParams.funNumCEC)

        specimen_val += self.penalty(specimen)
        trialVector_val += self.penalty(trialVector)

        global drawPlot
        if drawPlot:
            if self.dEParams.dimensions == 2:
                newValue = pd.DataFrame( [[specimen[0], specimen[1], specimen_val[0]]], columns=self.columns)
            else:
                newValue = pd.DataFrame( [[specimen[0], specimen_val[0]]], columns=self.columns)

            self.log = self.log.append(newValue, ignore_index=True)

            plt.cla()
            if self.dEParams.dimensions == 2:
                self.ax.scatter3D(self.log.x0, self.log.x1, self.log.y )
                self.ax.set_zlim3d(0,10000)
            else:
                self.ax.scatter(self.log.x0, self.log.y)
                self.ax.set_xlim(-100,100)
                self.ax.set_xlim(0,2000)

            self.fig.show()

        if trialVector_val <= specimen_val:
            population[np.where(population == specimen)[0][0]] = trialVector

    def run(self):
        global debug

        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        population = self.initialization()
        fes = 0

        while fes < params.maxfes:
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                self.evaluate(trialVector, specimen, population)
                fes += 1

            if (debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                y_val = [0]
                self.dEParams.evaluationFunction(randomSpecimen,
                                                 y_val, self.dEParams.dimensions, 1, self.dEParams.funNumCEC)
                print("Value: " + str(y_val) + "\n")

        self.dEParams.seed += 1
        return population


class DEANN(DE):

    def __init__(self, dEParams=DEParams(), dEANNParams=DEANNParams()):
        super().__init__(dEParams)
        self.dEANNParams = dEANNParams
        self.model, self.validation, self.evaluatedValidation = self.trainAndCompileModel()

    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.dEParams.dimensions, samples=self.dEANNParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(self.dEANNParams.trainingDataSize * 0.8)
        return data[:cut, :], data[cut:, :]

    # not used currently
    def dataNormalization(self, training, validation):
        normalizerTraining = Normalization(axis=-1)
        normalizerTraining.adapt(training)
        normalizerValidation = Normalization(axis=-1)
        normalizerValidation.adapt(validation)

        return normalizerTraining(training), normalizerValidation(validation)

    def evaluateSet(self, set):
        global evaluated
        evaluated = []
        for vector in set:
            val = [0]
            self.dEParams.evaluationFunction(vector, val, self.dEParams.dimensions, 1, self.dEParams.funNumCEC)
            evaluated.append(val)
        return np.array(evaluated)

    def evaluateWithModel(self, trialVector, specimen, population):
        specimen_val = self.model.predict(np.array(specimen).reshape(1, self.dEParams.dimensions))
        trialVector_val = self.model.predict(np.array(trialVector).reshape(1, self.dEParams.dimensions))

        specimen_val += self.penalty(specimen)
        trialVector_val += self.penalty(trialVector)

        global drawPlot
        if drawPlot:
            if self.dEParams.dimensions == 2:
                newValue = pd.DataFrame([[specimen[0], specimen[1], specimen_val[0]]], columns=self.columns)
            else:
                newValue = pd.DataFrame([[specimen[0], specimen_val[0]]], columns=self.columns)

            self.log = self.log.append(newValue)

            if self.dEParams.dimensions == 2:
                self.ax.scatter3D(self.log.x0, self.log.x1, self.log.y)
            else:
                self.ax.scatter(self.log.x0, self.log.y)

            self.fig.draw()

        if trialVector_val <= specimen_val:
                population[np.where(population == specimen)[0][0]] = trialVector

    def trainAndCompileModel(self):
        training, validation = self.generateTrainingData()

        evaluatedTraining = self.evaluateSet(training)
        evaluatedValidation = self.evaluateSet(validation)

        model = keras.Sequential()
        model.add(keras.Input(shape=(self.dEParams.dimensions)))

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

        model.fit(training, evaluatedTraining, epochs=self.dEANNParams.epochs,
                  validation_data=(validation, evaluatedValidation))
        return model, validation, evaluatedValidation

    def run(self):
        global debug

        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        population = self.initialization()
        fes = 0

        while fes < params.maxfes:
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                self.evaluateWithModel(trialVector, specimen, population)
                fes += 1

            if self.dEANNParams.teachModelEveryGeneration:
                evaluatedPopulation = self.evaluateSet(population)
                self.model.fit(population, evaluatedPopulation,
                               epochs=self.dEANNParams.teachModelEveryGenerationEpochs,
                               validation_data=(self.validation, self.evaluatedValidation))
            if (debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")

        self.dEParams.seed += 1
        return population


if __name__ == '__main__':

    usage = "usage: %prog [options]\n" \
            "Debug: -q\n" \
            "Params for DE: -s, -d, -f, -n, -c, -w, -m, -p\n" \
            "Params only for DEANN: -t, -e, -g, -r\n"
    parser = OptionParser(usage=usage)

    parser.add_option("-q", "--debug", action="store_true", dest="debug", default=False,
                      help="Prints debug info")
    parser.add_option("-i", "--iterations", type="int", dest="iterations", default=10,
                      help="Number of algorithms runs, incrementing seed by 1 (default 10)")
    parser.add_option("-l", "--plot", action="store_true", dest="plot", default=False,
                      help="Draw plot, works only for 1D and 2D.")
    parser.add_option("-s", "--seed", type="int", dest="seed", default=42,
                      help="Initial seed for numpy and random (default 42)")
    parser.add_option("-d", "--dimensions", type="int", dest="dimensions", default=2,
                      help="Number of dimension (default 2)")
    parser.add_option("-f", "--function", type="int", dest="function", default=31,
                      help="function number (default 31) [number 31 is quadratic function, number 4 is Rosenbrock]")
    parser.add_option("-n", "--popSize", type="int", dest="popSize", default=100,
                      help="Population size (default 100)")
    parser.add_option("-c", "--crossover", type="float", dest="crossoverProbability", default=0.7,
                      help="Crossover probability (default 0.7)")
    parser.add_option("-w", "--diffWeight", type="float", dest="differentialWeight", default=0.8,
                      help="Differential weight (default 0.8)")
    parser.add_option("-m", "--maxfes", type="int", dest="maxfes", default=2000,
                      help="Max function evaluations (default 2000)")
    parser.add_option("-p", "--penaltyFactor", type="float", dest="penaltyFactor", default=0.1,
                      help="Penalty factor (default 0.1)")
    parser.add_option("-t", "--trainingSize", type="int", dest="trainingSize", default=10000,
                      help="DEANN: training + validation data size (default 10000) [split is 80/20]")
    parser.add_option("-e", "--epochs", type="int", dest="epochs", default=10,
                      help="DEANN: epochs for model training (default 10)")
    parser.add_option("-g", "--teachGeneration", action="store_true", dest="teachGeneration", default=False,
                      help="DEANN: epochs for model training (default False)")
    parser.add_option("-r", "--epochsGeneration", type="int", dest="epochsGeneration", default=10,
                      help="DEANN: epochs for model training (default 10)")


    (options, args) = parser.parse_args()

    global debug
    debug = options.debug

    global drawPlot
    drawPlot = options.plot
    if options.dimensions != 1 and options.dimensions != 2:
        drawPlot = False

    params = DEParams(populationSize=options.popSize,
                      crossoverProbability=options.crossoverProbability,
                      differentialWeight=options.differentialWeight,
                      maxfes=options.maxfes,
                      dimensions=options.dimensions,
                      funNumCEC=options.function,
                      penaltyFactor=options.penaltyFactor,
                      seed=options.seed)

    dEAANParams = DEANNParams(trainingDataSize=options.trainingSize,
                              epochs=options.epochs,
                              teachModelEveryGeneration=options.teachGeneration,
                              teachModelEveryGenerationEpochs=options.epochsGeneration)

    iters = options.iterations

    DE_alg = DE(params)

    for i in range(iters):
        print("##### " + str(i + 1) + " #####")

        population = DE_alg.run()

        # find best specimen
        best = population[0]
        best_val = [0]
        params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
        for s in population:
            s_val = [0]
            params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
            if s_val <= best_val:
                best = s
                best_val = s_val

        if (debug):
            print("\nNajlepszy: " + str(best))
        print("Wartość: ")
        print(best_val)

    DEANN_alg = DEANN(params, dEAANParams)

    for i in range(iters):
        print("##### " + str(i + 1) + " #####")

        population = DEANN_alg.run()

        # find best specimen
        best = population[0]
        best_val = [0]
        params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
        for s in population:
            s_val = [0]
            params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
            if s_val <= best_val:
                best = s
                best_val = s_val

        if (debug):
            print("\nNajlepszy: " + str(best))
        print("Wartość: ")
        print(best_val)
