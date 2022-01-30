import random
import numpy as np
import pandas as pd
from cec21_functions import cec21_test_func
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
        self.evaluationFunction = cec21_test_func
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
                fes += 2

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
