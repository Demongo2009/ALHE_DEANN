from DE import *
from ANN import *

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