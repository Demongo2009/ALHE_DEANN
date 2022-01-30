
# IBHM configuration ------------------------------------------------------

ConfigureIBHM  <- function(stop.criterion.frac = 0.15,
                           stop.criterion.max.failures = 2,
                           stop.criterion.approximate.timeout = Inf,
                           weighting.function = function(y, w.par){ 0.01+dnorm(y,sd=abs(w.par))},
                           weighting.par = 1.0,
                           scal.candidates = c('dot.pr','root.radial'),                                                      
                           scal.lambda = 0.1,
                           activ.candidates = c('tanh','lin'),
                           optim.maxit = 500,
                           optim.retries = 1,
                           jit=TRUE,                           
                           verbose=FALSE,
                           final.estimation = 'all',
                           final.estimation.maxit = 20
){    
  optim.params = list(retries=optim.retries, maxit=optim.maxit, stopfitness = 0)
  #optim.params = list(retries=optim.retries, maxit=optim.maxit, stop.fitness = 0, CR = 0.5, F = 0.5, strategy = 2)
  
  optim = 'CMAES'
  
  list( unparsed.config = as.list(match.call()),
        sc.frac = stop.criterion.frac,
        sc.max.failures = stop.criterion.max.failures,
        sc.approximate.timeout = stop.criterion.approximate.timeout,
        wf = weighting.function,
        weighting.par = weighting.par,
        final.estimation = switch(final.estimation,
                                  weights = OptimizeAllWeights,
                                  all = OptimizeAllParams,
                                  stop('Unknow final.estimation type: ',final.estimation)),
        final.estimation.maxit = final.estimation.maxit,
        jit=jit,
        verbose=verbose,
        scal = list(                  
          optim = switch(optim, 
                         CMAES = OptimizeCMAES,
                         DE = OptimizeDE,
                         NM = OptimizeNM,
                         BFGS = OptimizeBFGS,
                         PSO  = OptimizePSO,                         
                         stop('Unknown optimization method: ',optim)),
          optim.params = optim.params,
          lambda = scal.lambda,
          candidates = ScalFunctions(scal.candidates)        
        ),
        activ = list(          
          optim = switch(optim, 
                         CMAES = OptimizeCMAES,
                         DE = OptimizeDE,
                         NM = OptimizeNM,
                         BFGS = OptimizeBFGS,
                         PSO  = OptimizePSO,
                         stop('Unknown optimization method:',optim)),
          optim.params = optim.params,
          candidates = ActivationCandidates(activ.candidates)
        )
  )
}