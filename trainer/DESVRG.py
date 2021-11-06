from pyDOE import lhs

from DE import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import heapq

class DESVRGParams:
    def __init__(self,
                 trainingDataSize=1000,
                 teachModel=True,
                 everyNGeneration=20,
                 kernel="rbf",
                 degree=5,
                 C=100,
                 gamma=0.1,
    ):
        """
        Params for DEANN version of algorithm especially ANN.


        :param teachModelEveryGeneration:
        :param teachModelEveryGenerationEpochs:
        """
        self.trainingDataSize = trainingDataSize
        self.teachModel = teachModel
        self.everyNGeneration = everyNGeneration
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.gamma = gamma


class DESVRG(DE):

    def __init__(self, dEParams=DEParams(), dESVRGParams=DESVRGParams()):
        super().__init__(dEParams)
        self.dESVRGParams = dESVRGParams
        self.model = self.trainAndCompileModel()


    def trainAndCompileModel(self):
        model = SVR(kernel=self.dESVRGParams.kernel)
        dict = {"C" : self.dESVRGParams.C, "gamma" : self.dESVRGParams.gamma}
        model.set_params(**dict)

        training, validation = self.generateTrainingData()
        evaluatedTraining = self.evaluateSet(training)
        evaluatedValidation = self.evaluateSet(validation)

        self.scX = StandardScaler()
        self.scy = StandardScaler()
        # training = self.scX.fit_transform(training)
        # evaluatedTraining = self.scy.fit_transform(evaluatedTraining)
        evaluatedTraining = [item for sublist in evaluatedTraining for item in sublist]

        model.fit(training, evaluatedTraining)
        return model



    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.dEParams.dimensions, samples=self.dESVRGParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(self.dESVRGParams.trainingDataSize * 0.8)
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

    def evaluateAndReturn(self, trialVector, specimen, population):
        specimen_val = [0]
        trialVector_val = [0]
        self.dEParams.evaluationFunction(specimen, specimen_val, self.dEParams.dimensions, 1, self.dEParams.funNumCEC)
        self.dEParams.evaluationFunction(trialVector, trialVector_val, self.dEParams.dimensions, 1,
                                         self.dEParams.funNumCEC)

        specimen_val += self.penalty(specimen)
        trialVector_val += self.penalty(trialVector)

        if trialVector_val <= specimen_val:
            population[np.where(population == specimen)[0][0]] = trialVector

        return specimen_val, trialVector_val


    def run(self):

        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        population = self.initialization()
        fes = self.dESVRGParams.trainingDataSize
        generation = 0
        while fes < self.dEParams.maxfes:
            originalValuesForMSE = []
            modelValuesForMSE = []
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                self.evaluateWithModel(trialVector, specimen, population)
                # specimen_val, trialVector_val = self.evaluateAndReturn(trialVector, specimen, population)
                # originalValuesForMSE.append(specimen_val)
                # originalValuesForMSE.append(trialVector_val)
                #
                # specimen_val = self.model.predict(np.array(specimen).reshape(1, self.dEParams.dimensions))
                # trialVector_val = self.model.predict(np.array(trialVector).reshape(1, self.dEParams.dimensions))
                #
                # specimen_val += self.penalty(specimen)
                # trialVector_val += self.penalty(trialVector)
                #
                # modelValuesForMSE.append(specimen_val)
                # modelValuesForMSE.append(trialVector_val)

            if self.dESVRGParams.teachModel and generation % 20 == 0:
                selectedPopulation = self.selectMostDistant(population)
                evaluatedPopulation = self.evaluateSet(selectedPopulation)
                fes += 20

                # selectedPopulation = self.scX.transform(selectedPopulation)
                # evaluatedPopulation = self.scy.transform(evaluatedPopulation)
                evaluatedPopulation = [item for sublist in evaluatedPopulation for item in sublist]

                self.dESVRGParams.C*=1.01
                self.dESVRGParams.gamma*=0.99
                dict = {"C": self.dESVRGParams.C, "gamma": self.dESVRGParams.gamma}
                self.model.set_params(**dict)

                self.model.fit(selectedPopulation, evaluatedPopulation)
                print("Number of SVs: " + str(len(self.model.support_vectors_)))
            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")

            # originalValuesForMSE = self.scy.transform(originalValuesForMSE)
            # indices = random.sample(range(200),20)
            # originalValuesForMSE = [originalValuesForMSE[i] for i in indices]
            # modelValuesForMSE = [modelValuesForMSE[i] for i in indices]
            # MSE = mean_squared_error(originalValuesForMSE, modelValuesForMSE)
            # print("MSE: " + str(MSE))
            generation += 1
        self.dEParams.seed += 1
        return population

    def selectMostDistant(self, population):
        listDistances = []
        svs = self.model.support_vectors_

        for i,specimen in enumerate(population):
            listForSV = [0]
            for sv in svs:
                distance = 0
                for j in range(self.dEParams.dimensions):
                    distance += (sv[j] - specimen[j])**2
                distance = distance**(1/2)
                listForSV.append(distance)

            listDistances.append((min(listForSV),i))
        listDistances = [(value*-1, i) for (value, i) in listDistances]
        heapq.heapify(listDistances)
        selected = []
        for i in range(20):
            (value, index) = heapq.heappop(listDistances)
            selected.append(population[index*-1])

        return selected