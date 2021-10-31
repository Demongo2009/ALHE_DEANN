from DE import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class DESVRGParams:
    def __init__(self, trainingDataSize=1000,
                 teachModelEveryGeneration=False,
                 teachModelEveryGenerationEpochs=10):
        """
        Params for DEANN version of algorithm especially ANN.


        :param teachModelEveryGeneration:
        :param teachModelEveryGenerationEpochs:
        """


        self.teachModelEveryGeneration = teachModelEveryGeneration
        self.teachModelEveryGenerationEpochs = teachModelEveryGenerationEpochs


class DESVRG(DE):

    def __init__(self, dEParams=DEParams(), dESVRGParams=DESVRGParams()):
        super().__init__(dEParams)
        self.dESVRGParams = dESVRGParams
        self.model = trainAndCompileModel()

    def trainAndCompileModel(self):
        model = SVR(kernel='rbf')

        training, validation = generateTrainingData()
        evaluatedTraining = self.evaluateSet(training)
        evaluatedValidation = self.evaluateSet(validation)

        scX = StandardScaler()
        scy = StandardScaler()
        training = scX.fit_transform(training)
        evaluatedTraining = scy.fit_transform(evaluatedTraining)

        model.fit(training, evaluatedTraining)
        return model



    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.aNNParams.dimensions, samples=self.dEParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(self.aNNParams.trainingDataSize * 0.8)
        return data[:cut, :], data[cut:, :]

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