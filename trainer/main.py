
from DE import *
from DEANN import *
from DESVR import *
from optparse import OptionParser

from DEIBHM import *

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

    # dEAANParams = DEANNParams(teachModelEveryGeneration=options.teachGeneration,
    #                           teachModelEveryGenerationEpochs=options.epochsGeneration)
    #
    # aNNParams = ANNParams(dimensions=options.dimensions,
    #                       trainingDataSize=options.trainingSize,
    #                         epochs=options.epochs,
    #                       numberOfHiddenLayers=5,
    #                       numberOfStartNeurons=500,
    #                       learningRate=3e-8,
    #                       useCV=True,
    #                       funNumCEC=options.function
    #                       )

    dESVRGParams = DESVRParams(
        trainingDataSize=1000,
        teachModel=True,
        everyNGeneration=20,
        kernel="rbf",
        C=100,
        gamma=100,
    )

    dESVRPParams = DESVRParams(
        trainingDataSize=1000,
        teachModel=True,
        everyNGeneration=20,
        kernel="poly",
        degree=5,
        C=10,
    )

    dEIBHMarams = DEIBHMParams(
        trainingDataSize=1000,
        teachModel=True,
        everyNGeneration=20,
    )

    iters = options.iterations



    # DE_alg = DE(params)
    #
    # for i in range(iters):
    #     print("##### " + str(i + 1) + " #####")
    #
    #     population = DE_alg.run()
    #
    #     # find best specimen
    #     best = population[0]
    #     best_val = [0]
    #     params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
    #     for s in population:
    #         s_val = [0]
    #         params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
    #         if s_val <= best_val:
    #             best = s
    #             best_val = s_val
    #
    #     if (debug):
    #         print("\nNajlepszy: " + str(best))
    #     print("Wartość: ")
    #     print(best_val)



    # DESVRG_alg = DESVR(params, dESVRGParams)
    #
    # for i in range(iters):
    #     print("##### DESVRG " + str(i + 1) + " #####")
    #
    #     population = DESVRG_alg.run()
    #
    #     # find best specimen
    #     best = population[0]
    #     best_val = [0]
    #     params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
    #     for s in population:
    #         s_val = [0]
    #         params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
    #         if s_val <= best_val:
    #             best = s
    #             best_val = s_val
    #
    #     if (debug):
    #         print("\nNajlepszy: " + str(best))
    #     print("Wartość: ")
    #     print(best_val)



    # DESVRP_alg = DESVR(params, dESVRPParams)
    #
    # for i in range(iters):
    #     print("##### DESVRP " + str(i + 1) + " #####")
    #
    #     population = DESVRP_alg.run()
    #
    #     # find best specimen
    #     best = population[0]
    #     best_val = [0]
    #     params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
    #     for s in population:
    #         s_val = [0]
    #         params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
    #         if s_val <= best_val:
    #             best = s
    #             best_val = s_val
    #
    #     if (debug):
    #         print("\nNajlepszy: " + str(best))
    #     print("Wartość: ")
    #     print(best_val)


    DEIBHM_alg = DEIBHM(params, dEIBHMarams)

    for i in range(iters):
        print("##### DESVRP " + str(i + 1) + " #####")

        population = DEIBHM_alg.run()

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


    # DEANN_alg = DEANN(params, dEAANParams, aNNParams)
    #
    # for i in range(iters):
    #     print("##### DEANN " + str(i + 1) + " #####")
    #
    #     population = DEANN_alg.run()
    #
    #     # find best specimen
    #     best = population[0]
    #     best_val = [0]
    #     params.evaluationFunction(best, best_val, params.dimensions, 1, params.funNumCEC)
    #     for s in population:
    #         s_val = [0]
    #         params.evaluationFunction(s, s_val, params.dimensions, 1, params.funNumCEC)
    #         if s_val <= best_val:
    #             best = s
    #             best_val = s_val
    #
    #     if (debug):
    #         print("\nNajlepszy: " + str(best))
    #     print("Wartość: ")
    #     print(best_val)
