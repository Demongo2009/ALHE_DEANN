
from optparse import OptionParser


import os
import time
import random
from pyDOE import lhs
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

from ctypes import CDLL, POINTER, c_int, c_double
import os
import platform
import numpy as np


if platform.system() == "Windows":
    lib = '/content/drive/My Drive/cec17_test_func_win.so'
elif platform.system() == "Linux":
    lib = '/content/drive/My Drive/cec17_test_func_lin.so'
else:
    lib = 'Something wrong'

def cec17_test_func(x, f, nx, mx, func_num,
                    dll_path=CDLL(os.path.abspath(lib))):

    if func_num == 31:
        f[0] = sum([value ** 2 for value in x])
        return

    # if func_num == 32:
    #     f[0] = sum(
    #         100.0 *
    #         np.power(
    #             (np.subtract(x[1:], np.power(x[:-1], 2))), 2)
    #         + np.power((np.subtract(1, x[:-1])), 2))
    #     return

    functions = dll_path
    x_pointer_type = POINTER(c_double * nx)
    f_pointer_type = POINTER(c_double * mx)
    nx_type = c_int
    mx_type = c_int
    func_num_type = c_int
    functions.cec17_test_func.argtypes = [x_pointer_type, f_pointer_type,
                                          nx_type, mx_type, func_num_type]
    functions.cec17_test_func.restype = None
    x_ctype = (c_double * nx)()
    for i, value in enumerate(x):
        x_ctype[i] = value
    f_ctype = (c_double * mx)()
    for i in range(mx):
        f_ctype[i] = 0
    functions.cec17_test_func(x_pointer_type(x_ctype), f_pointer_type(f_ctype),
                              nx, mx, func_num)
    for i in range(len(f)):
        f[i] = f_ctype[i]



class ANNParams:
    def __init__(self, dimensions = 2, trainingDataSize=10000, epochs=10,
                 numberOfHiddenLayers = 3, numberOfStartNeurons=100, learningRate = 3e-6, useCV=False,
                 funNumCEC = 1):
        """
        Params for ANN trainer created for ANN usage in DEANN. Can also be used to train and save ANN.

        :param dimensions:
        :param trainingDataSize:
        :param epochs:
        :param numberOfHiddenLayers:
        :param numberOfStartNeurons: number of neurons, on which program will start hidden layers then
        decreasing by half
        """
        self.dimensions = dimensions

        self.trainingDataSize = trainingDataSize
        self.epochs = epochs

        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.numberOfStartNeurons = numberOfStartNeurons

        self.useCV = useCV
        self.evaluationFunction = cec17_test_func
        self.funNumCEC = funNumCEC
        self.learningRate = learningRate

class ANNTrainer():
    def __init__(self, aNNParams = ANNParams()):
        self.aNNParams = aNNParams

    def evaluateSet(self, set):
        global evaluated
        evaluated = []
        for vector in set:
            val = [0]
            self.aNNParams.evaluationFunction(vector, val, self.aNNParams.dimensions, 1, self.aNNParams.funNumCEC)
            evaluated.append(val)
        return np.array(evaluated)

    def trainAndCompileModel(self):
        training, validation = self.generateTrainingData()
        evaluatedTraining = self.evaluateSet(training)
        evaluatedValidation = self.evaluateSet(validation)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True)

        root_logdir = os.path.join(os.curdir, "../my_logs")
        def get_run_logdir():
            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            return os.path.join(root_logdir, run_id)

        run_logdir = get_run_logdir()
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


        if self.aNNParams.useCV:
            keras_reg = keras.wrappers.scikit_learn.KerasRegressor(self.buildModel)
            param_distribs = {
                "n_hidden": np.arange(1, self.aNNParams.numberOfHiddenLayers).tolist(),
                "n_neurons": np.arange(1, self.aNNParams.numberOfStartNeurons).tolist(),
                "learning_rate": np.arange(3e-6, 3e-2).tolist(),
            }
            rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
            rnd_search_cv.fit(training, evaluatedTraining, epochs=self.aNNParams.epochs,
                  validation_data=(validation, evaluatedValidation),
                  callbacks=[early_stopping_cb, tensorboard_cb])
            print(rnd_search_cv.best_params_)
            model = rnd_search_cv.best_estimator_.model
        else:
            model = self.buildModel(n_hidden= self.aNNParams.numberOfHiddenLayers,
                                    n_neurons= self.aNNParams.numberOfStartNeurons,
                                    learning_rate= self.aNNParams.learningRate)
            model.fit(training, evaluatedTraining, epochs=self.aNNParams.epochs,
                  validation_data=(validation, evaluatedValidation),
                  callbacks=[early_stopping_cb, tensorboard_cb])


        return model, validation, evaluatedValidation


    def buildModel(self, n_hidden = 3, n_neurons = 20, learning_rate = 3e-2):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.aNNParams.dimensions)))


        for i in range(n_hidden):
            model.add(keras.layers.BatchNormalization())
            model.add(layers.Dropout(0.2))
            model.add(
                layers.Dense(n_neurons, activation=keras.layers.LeakyReLU(alpha=0.2),
                             kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
                             kernel_initializer=keras.initializers.he_uniform))
            n_neurons/=2

        model.add(keras.layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(1, activation=keras.activations.linear,
                         kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))

        model.compile(optimizer=keras.optimizers.Nadam(clipnorm=1, learning_rate=learning_rate),
                      loss=keras.losses.MeanSquaredError())
        return model




    # not used currently
    def dataNormalization(self, training, validation):
        normalizerTraining = Normalization(axis=-1)
        normalizerTraining.adapt(training)
        normalizerValidation = Normalization(axis=-1)
        normalizerValidation.adapt(validation)

        return normalizerTraining(training), normalizerValidation(validation)


    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.aNNParams.dimensions, samples=self.aNNParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(self.aNNParams.trainingDataSize * 0.8)
        return data[:cut, :], data[cut:, :]



import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DEParams:
    def __init__(self, populationSize=100, crossoverProbability=0.7, differentialWeight=0.8, maxfes=2000,
                 dimensions=5, minValue=-100, maxValue=100, funNumCEC=1, penaltyFactor=0.1, seed=42,
                 drawPlot=False, debug=False):
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
        self.drawPlot = drawPlot
        self.debug = debug


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
            donorVector[i] if random.uniform(0, 1) <= self.dEParams.crossoverProbability or i == randomI
            else specimen[i]
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


        if self.dEParams.drawPlot:
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

        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        population = self.initialization()
        fes = 0

        while fes < self.dEParams.maxfes:
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                self.evaluate(trialVector, specimen, population)
                fes += 1

            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                y_val = [0]
                self.dEParams.evaluationFunction(randomSpecimen,
                                                 y_val, self.dEParams.dimensions, 1, self.dEParams.funNumCEC)
                print("Value: " + str(y_val) + "\n")

        self.dEParams.seed += 1
        return population


class DEANNParams:
    def __init__(self, teachModelEveryGeneration=False,
                 teachModelEveryGenerationEpochs=10):
        """
        Params for DEANN version of algorithm especially ANN.


        :param teachModelEveryGeneration:
        :param teachModelEveryGenerationEpochs:
        """


        self.teachModelEveryGeneration = teachModelEveryGeneration
        self.teachModelEveryGenerationEpochs = teachModelEveryGenerationEpochs


class DEANN(DE):

    def __init__(self, dEParams=DEParams(), dEANNParams=DEANNParams(), aNNParams=ANNParams()):
        super().__init__(dEParams)
        self.dEANNParams = dEANNParams
        self.aNNTrainer = ANNTrainer(aNNParams)
        self.model, self.validation, self.evaluatedValidation = self.aNNTrainer.trainAndCompileModel()


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

        if self.dEParams.drawPlot:
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


    def run(self):

        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        population = self.initialization()
        fes = 0

        while fes < self.dEParams.maxfes:
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
            if (self.dEParams.debug):
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

    debug = options.debug

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
                      seed=options.seed,
                      drawPlot=drawPlot)

    dEAANParams = DEANNParams(teachModelEveryGeneration=options.teachGeneration,
                              teachModelEveryGenerationEpochs=options.epochsGeneration)

    aNNParams = ANNParams(dimensions=options.dimensions,
                          trainingDataSize=options.trainingSize,
                            epochs=options.epochs,
                          numberOfHiddenLayers=1,
                          numberOfStartNeurons=10000,
                          learningRate=3e-6,
                          useCV=False,
                          funNumCEC=options.function
                          )

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

    DEANN_alg = DEANN(params, dEAANParams, aNNParams)

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
