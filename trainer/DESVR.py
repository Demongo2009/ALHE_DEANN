from pyDOE import lhs

from DE import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import heapq

class DESVRParams:
    def __init__(self,
                 trainingDataSize=1000,
                 teachModel=True,
                 everyNGeneration=20,
                 kernel="rbf",
                 degree=5,
                 C=0.1,
                 gamma=100,
                 useTrainingSet=False,
    ):
        """
        Params for DESVR version of algorithm.


        :param trainingDataSize: size of generated and evaluated training data
        :param teachModel: is the model supposed to refit
        :param everyNGeneration: use refit every N generation
        :param kernel: kernel for SVR, in my case rbf or poly
        :param degree: degree of poly kernel
        :param C: regularization param
        :param gamma: parameter for gaussian rbf kernel, impacting width of the curve
        :param useTrainingSet: is the model supposed to train on whole training set
        """
        self.trainingDataSize = trainingDataSize
        self.teachModel = teachModel
        self.everyNGeneration = everyNGeneration
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.gamma = gamma
        self.useTrainingSet = useTrainingSet


class DESVR(DE):

    def __init__(self, dEParams=DEParams(), dESVRParams=DESVRParams()):
        super().__init__(dEParams)
        self.dESVRParams = dESVRParams


    def trainAndCompileModel(self):
        model = SVR(kernel=self.dESVRParams.kernel)
        dict = {}
        if self.dESVRParams.kernel == "rbf":
            dict = {"C" : self.dESVRParams.C, "gamma" : self.dESVRParams.gamma}
        else:
            if self.dESVRParams.kernel == "poly":
                dict = {"C": self.dESVRParams.C, "degree": self.dESVRParams.degree}
        model.set_params(**dict)

        # training, validation = self.generateTrainingData()
        training= self.generateTrainingData()
        evaluatedTraining = self.evaluateSet(training)
        # evaluatedValidation = self.evaluateSet(validation)

        evaluatedTraining = [item for sublist in evaluatedTraining for item in sublist]
        self.trainingEvaluatedPairSet = [ (item, evaluatedTraining[i]) for i,item in enumerate(training)]

        model.fit(training, evaluatedTraining)
        return model



    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.dEParams.dimensions, samples=self.dESVRParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        cut = np.int32(self.dESVRParams.trainingDataSize * 0.8)
        # return data[:cut, :], data[cut:, :]
        return data

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


    def runEX0(self):
        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)

        self.C = self.dESVRParams.C
        self.gamma = self.dESVRParams.gamma

        self.model = self.trainAndCompileModel()

        population = self.initialization()
        fes = self.dESVRParams.trainingDataSize
        generation = 0
        while fes < self.dEParams.maxfes:
            originalValuesForMSE = []
            modelValuesForMSE = []
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                specimen_val, trialVector_val = self.evaluateAndReturn(trialVector, specimen, population)
                originalValuesForMSE.append(specimen_val)
                originalValuesForMSE.append(trialVector_val)

                specimen_val = self.model.predict(np.array(specimen).reshape(1, self.dEParams.dimensions))
                trialVector_val = self.model.predict(np.array(trialVector).reshape(1, self.dEParams.dimensions))

                specimen_val += self.penalty(specimen)
                trialVector_val += self.penalty(trialVector)

                modelValuesForMSE.append(specimen_val)
                modelValuesForMSE.append(trialVector_val)

            if self.dESVRParams.teachModel and generation % 20 == 0:
                selectedPopulation = self.selectMostDistant(population)
                evaluatedPopulation = self.evaluateSet(selectedPopulation)
                fes += 20

                evaluatedPopulation = [item for sublist in evaluatedPopulation for item in sublist]

                dict = {}
                if self.dESVRParams.kernel == "rbf":
                    self.C *= 0.95
                    self.gamma *= 0.95

                    if self.C <= 0.01:
                        self.C = 0.01

                    if self.gamma <= 0.01:
                        self.gamma = 0.01
                    dict = {"C": self.C, "gamma": self.gamma}
                else:
                    if self.dESVRParams.kernel == "poly":
                        self.C *= 0.95

                        if self.C <= 0.01:
                            self.C = 0.01
                        dict = {"C": self.C, "degree": self.dESVRParams.degree}

                self.model.set_params(**dict)

                if self.dESVRParams.useTrainingSet:
                    self.updateTrainingSet(selectedPopulation, evaluatedPopulation)
                    training = [ x for (x, _) in self.trainingEvaluatedPairSet]
                    evaluatedTraining = [ x for (_, x) in self.trainingEvaluatedPairSet]
                else:
                    training = selectedPopulation
                    evaluatedTraining = evaluatedPopulation

                self.model.fit(training, evaluatedTraining)



            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")

            if generation % 100 == 0:
                MSE = mean_squared_error(originalValuesForMSE, modelValuesForMSE)
                print("Generation: " + str(generation))
                print("MSE: " + str(MSE))
                print("Number of SVs: " + str(len(self.model.support_vectors_))+ "\n")
            generation += 1
        self.dEParams.seed += 1
        # print(self.dEParams.seed)
        return population

    def run(self):
        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)


        self.C = self.dESVRParams.C
        self.gamma = self.dESVRParams.gamma

        self.model = self.trainAndCompileModel()


        population = self.initialization()
        fes = self.dESVRParams.trainingDataSize
        generation = 0
        while fes < self.dEParams.maxfes:
            for specimen in population:
                indexOfSpecimen = np.where(population == specimen)[0][0]

                individuals = self.generate(population, indexOfSpecimen)
                donorVector = self.mutation(individuals)
                trialVector = self.crossover(specimen, donorVector)

                self.evaluateWithModel(trialVector, specimen, population)

            if self.dESVRParams.teachModel and generation % 20 == 0:
                selectedPopulation = self.selectMostDistant(population)
                evaluatedPopulation = self.evaluateSet(selectedPopulation)
                fes += 20

                evaluatedPopulation = [item for sublist in evaluatedPopulation for item in sublist]

                # evaluatedPopulation = [item if np.isfinite(item) else 100000 for item in evaluatedPopulation]


                dict = {}
                if self.dESVRParams.kernel == "rbf":
                    self.C *= 0.95
                    self.gamma *= 0.95

                    if self.C <= 0.01:
                        self.C = 0.01

                    if self.gamma <= 0.01:
                        self.gamma = 0.01
                    dict = {"C": self.C, "gamma": self.gamma}
                else:
                    if self.dESVRParams.kernel == "poly":
                        self.C *= 0.95

                        if self.C <= 0.001:
                            self.C = 0.001
                        dict = {"C": self.C, "degree": self.dESVRParams.degree}
                self.model.set_params(**dict)

                # print(self.C)
                # print(self.gamma)
                # print(evaluatedPopulation)

                if self.dESVRParams.useTrainingSet:
                    self.updateTrainingSet(selectedPopulation, evaluatedPopulation)
                    training = [ x for (x, _) in self.trainingEvaluatedPairSet]
                    evaluatedTraining = [ x for (_, x) in self.trainingEvaluatedPairSet]
                else:
                    training = selectedPopulation
                    evaluatedTraining = evaluatedPopulation

                self.model.fit(training, evaluatedTraining)

            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")
            # if generation % 10000 == 0:
            #     print("Generation: " + str(generation))
            #     print("Number of SVs: " + str(len(self.model.support_vectors_)))
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

    def updateTrainingSet(self, newTrainingPoints, evaluatedNewTrainingPoints):
        sortedTrainingSet = sorted(self.trainingEvaluatedPairSet, key=lambda x: x[1])[:self.dESVRParams.trainingDataSize-20]

        for i,item in enumerate(newTrainingPoints):
            sortedTrainingSet.append((item, evaluatedNewTrainingPoints[i]))

        self.trainingEvaluatedPairSet = sortedTrainingSet






