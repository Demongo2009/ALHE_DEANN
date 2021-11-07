import os
import time
from pyDOE import lhs
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
from trainer.cec21_functions import cec21_test_func
from sklearn.model_selection import RandomizedSearchCV


class ANNParams:
    def __init__(self, dimensions = 2, trainingDataSize=10000, epochs=10,
                 numberOfHiddenLayers = 3, numberOfStartNeurons=100, learningRate = 3e-6, useCV=False,
                 funNumCEC = 1, loadModel = "./my_models", pathToModel = "./my_models", saveToPath = "./my_models"):
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
        self.evaluationFunction = cec21_test_func
        self.funNumCEC = funNumCEC
        self.learningRate = learningRate
        self.loadModel = loadModel
        self.pathToModel = pathToModel
        self.saveToPath = saveToPath

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

        if(self.aNNParams.loadModel):
            return keras.models.load_model(self.aNNParams.pathToModel), validation, evaluatedValidation

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

        model.save(self.aNNParams.saveToPath + time.strftime("%Y_%m_%d-%H_%M_%S"))
        return model, validation, evaluatedValidation


    def buildModel(self, n_hidden = 3, n_neurons = 20, learning_rate = 3e-3, centers=""):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.aNNParams.dimensions)))
        if centers == "":
            centers = np.loadtxt("rbf_for_tf2/data/data.txt")[:, :-1]

        for i in range(n_hidden):
            model.add(keras.layers.BatchNormalization())
            model.add(layers.Dropout(0.2))
            model.add(
                layers.Dense(n_neurons, activation=keras.activations.relu,
                             kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
                             kernel_initializer=keras.initializers.he_uniform()))
            n_neurons/=2

        #         rbflayer = RBFLayer(n_neurons, betas=4.0, input_shape=(
        #                    self.aNNParams.dimensions,), initializer=InitCentersKMeans(centers) )
        # model.add(rbflayer)

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