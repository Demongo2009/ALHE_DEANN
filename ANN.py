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
from cec17_functions import cec17_test_func
import matplotlib.pyplot as plt

class ANNParams:
    def __init__(self, dimensions = 2, trainingDataSize=10000, epochs=10,
                 numberOfHiddenLayers = 3, numberOfStartNeurons=100, useCV=False,
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

        root_logdir = os.path.join(os.curdir, "my_logs")

        def get_run_logdir():
            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            return os.path.join(root_logdir, run_id)

        run_logdir = get_run_logdir()

        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

        # TODO: chech this, fix
        if self.aNNParams.useCV:
            keras_reg = keras.wrappers.scikit_learn.KerasRegressor(self.buildModel)
            model = keras_reg
        else:
            model = self.buildModel()

        model.fit(training, evaluatedTraining, epochs=self.aNNParams.epochs,
                  validation_data=(validation, evaluatedValidation),
                  callbacks=[early_stopping_cb, tensorboard_cb])
        return model, validation, evaluatedValidation


    def buildModel(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.aNNParams.dimensions)))

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