
# Differential Evolution algorithm using Artificial Neural Network  

## Authors  
Jakub Gałat  
Konrad Kulesza  

## Description

Differential Evolution with a surrogate model aimed at cost function approximation. The task involves designing and implementing a surrogate model, which will in a adequate manner approximate cost function in the Differential Evolution algorithm. New specimen evaluation will use developed surrogate model, which will lead to reduced overhead connected to real cost function evaluation. The task also involves the basic Differential Evolution (without surrogate model) comparison.

## Work summary

We focused on analysis of Differential Evolution algorithm, both with and without surrogate model. Modified algorithm is compared to original by changing one of the parameters of both algorithms and comparing the averaged results. The experiments where conducted using quadratic function and Rosenbrock function. Due to not satisfying results of function approximation during tests on CEC2017 functions in 10 dimensions, we decided to halt experiments in that direction at the moment.

## Problem description

The most resource-hungry stage of Differential Evolution algorithm is specimen evaluation using cost function. The use of surrogate model mitigates the overhead required to evaluate (often complicated) functions describing some real phenomenon. Hence implementation of such model is a considerable option, however it comes with a price. It is often bound to the fact that it only approximates the real function, so the quality of results may differ (however sometimes it can be even better). 

## Solution description

Solution is implemented in Python, using Keras module. The Differential Evolution (DE) design follows written below:

![differentialEvolution](./assets/differentialEvolution.jpg)

In the case of Differential Evolution using Artificial Neural Network (DEANN), right after the *initialization* comes the stage of learning the neural network. The model is then used in *evaluate*, where it replaces original cost function. Influenced by the article ( [source](https://www.researchgate.net/publication/342548581_Using_Neural_Networks_as_Surrogate_Models_in_Differential_Evolution_Optimization_of_Truss_Structures) ), we decided that our neural network will have the *numberOfDimensions*-20-20-20-1 architecture. As a activation function we use ReLU in hidden layers.

Random number generation using Latin Hypercube Sampling ([wiki](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)) from module pyDOE.

## Usage
```shell
./main.py [options]

Options:
  -h, --help            show this help message and exit
  -q, --debug           Prints debug info
  -i ITERATIONS, --iterations=ITERATIONS
                        Number of algorithms runs, incrementing seed by 1
                        (default 10)
  -l, --plot            Draw plot, works only for 1D and 2D.
  -s SEED, --seed=SEED  Initial seed for numpy and random (default 42)
  -d DIMENSIONS, --dimensions=DIMENSIONS
                        Number of dimension (default 2)
  -f FUNCTION, --function=FUNCTION
                        function number (default 31) [number 31 is quadratic
                        function, number 4 is Rosenbrock]
  -n POPSIZE, --popSize=POPSIZE
                        Population size (default 100)
  -c CROSSOVERPROBABILITY, --crossover=CROSSOVERPROBABILITY
                        Crossover probability (default 0.7)
  -w DIFFERENTIALWEIGHT, --diffWeight=DIFFERENTIALWEIGHT
                        Differential weight (default 0.8)
  -m MAXFES, --maxfes=MAXFES
                        Max function evaluations (default 2000)
  -p PENALTYFACTOR, --penaltyFactor=PENALTYFACTOR
                        Penalty factor (default 0.1)
  -t TRAININGSIZE, --trainingSize=TRAININGSIZE
                        DEANN: training + validation data size (default 10000)
                        [split is 80/20]
  -e EPOCHS, --epochs=EPOCHS
                        DEANN: epochs for model training (default 10)
  -g, --teachGeneration
                        DEANN: epochs for model training (default False)
  -r EPOCHSGENERATION, --epochsGeneration=EPOCHSGENERATION
                        DEANN: epochs for model training (default 10)
```
