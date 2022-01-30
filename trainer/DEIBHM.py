import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import random

from pyDOE import lhs

from DE import *
from sklearn.metrics import mean_squared_error
import heapq

# import rpy2's package module
import rpy2.robjects.packages as rpackages



class DEIBHMParams:
    def __init__(self,
                 trainingDataSize=1000,
                 teachModel=True,
                 everyNGeneration=20,
    ):
        """
        Params for DESVR version of algorithm.


        :param trainingDataSize: size of generated and evaluated training data
        :param teachModel: is the model supposed to refit
        :param everyNGeneration: use refit every N generation
        """
        self.trainingDataSize = trainingDataSize
        self.teachModel = teachModel
        self.everyNGeneration = everyNGeneration


class DEIBHM(DE):

    def __init__(self, dEParams=DEParams(), dEIBHMParams=DEIBHMParams()):
        super().__init__(dEParams)
        self.dESVRParams = dEIBHMParams

        # robjects.r('install.packages("lubridate")')
        robjects.r(
            # '''
#             # source('R/scals.R')
# # source('R/activs.R')
# # source('R/optim.R')
# # source('R/stopCriteria.R')
# # source('R/config.R')
#
# # Activation functions ##############################
# ActivationCandidates <- function(filter = NULL){
#   # tanh activation function
#   Tanh <- function(x){
#     return(tanh(x));
#   }
#   expr(Tanh) <- 'tanh'
#
#
#   # logistic  activation function
#   LogSig <- function(x){
#     return(1/(1+exp(x)));
#   }
#   expr(LogSig) <- 'logsig'
#
#
#   # linear  activation function
#   Lin <- function(x){
#     return(x);
#   }
#   expr(Lin) <- 'lin'
#
#   res <- list(Tanh, LogSig, Lin)
#
#   if(!is.null(filter)){
#     res <- Filter(function(v) any(expr(v)==filter), res)
#   }
#   if(length(res)==0){
#     stop('Invalid activ list given : ', paste(filter,sep=', '))
#   }
#   res
# }
#
#
#
#
# # IBHM configuration ------------------------------------------------------
#
# ConfigureIBHM  <- function(stop.criterion.frac = 0.15,
#                            stop.criterion.max.failures = 2,
#                            stop.criterion.approximate.timeout = Inf,
#                            weighting.function = function(y, w.par){ 0.01+dnorm(y,sd=abs(w.par))},
#                            weighting.par = 1.0,
#                            scal.candidates = c('dot.pr','root.radial'),
#                            scal.lambda = 0.1,
#                            activ.candidates = c('tanh','lin'),
#                            optim.maxit = 500,
#                            optim.retries = 1,
#                            jit=TRUE,
#                            verbose=FALSE,
#                            final.estimation = 'all',
#                            final.estimation.maxit = 20
# ){
#   optim.params = list(retries=optim.retries, maxit=optim.maxit, stopfitness = 0)
#   #optim.params = list(retries=optim.retries, maxit=optim.maxit, stop.fitness = 0, CR = 0.5, F = 0.5, strategy = 2)
#
#   optim = 'CMAES'
#
#   list( unparsed.config = as.list(match.call()),
#         sc.frac = stop.criterion.frac,
#         sc.max.failures = stop.criterion.max.failures,
#         sc.approximate.timeout = stop.criterion.approximate.timeout,
#         wf = weighting.function,
#         weighting.par = weighting.par,
#         final.estimation = switch(final.estimation,
#                                   weights = OptimizeAllWeights,
#                                   all = OptimizeAllParams,
#                                   stop('Unknow final.estimation type: ',final.estimation)),
#         final.estimation.maxit = final.estimation.maxit,
#         jit=jit,
#         verbose=verbose,
#         scal = list(
#           optim = switch(optim,
#                          CMAES = OptimizeCMAES,
#                          DE = OptimizeDE,
#                          NM = OptimizeNM,
#                          BFGS = OptimizeBFGS,
#                          PSO  = OptimizePSO,
#                          stop('Unknown optimization method: ',optim)),
#           optim.params = optim.params,
#           lambda = scal.lambda,
#           candidates = ScalFunctions(scal.candidates)
#         ),
#         activ = list(
#           optim = switch(optim,
#                          CMAES = OptimizeCMAES,
#                          DE = OptimizeDE,
#                          NM = OptimizeNM,
#                          BFGS = OptimizeBFGS,
#                          PSO  = OptimizePSO,
#                          stop('Unknown optimization method:',optim)),
#           optim.params = optim.params,
#           candidates = ActivationCandidates(activ.candidates)
#         )
#   )
# }
#
# MultiOptim <- function(retries, optim.fun, ...){
#   best <- list(eval=Inf)
#   for(i in 1:retries){
#     res <- optim.fun(...)
#     if(res$eval < best$eval){
#       best <- res
#     }
#   }
#
#   res
# }
#
# # Function optimization ---------------------------------------
#
# OptimizeCMAES <- function(goal, n, optim.params){
#   MultiOptim(optim.params$retries, function(){
#       res <- cmaes::cma_es(runif(n, min=optim.params$par.min, max=optim.params$par.max),
#                            goal,
#                            lower=rep(optim.params$par.min,n),
#                            upper=rep(optim.params$par.max,n),
#                            control=list(maxit=optim.params$maxit,
#                                         stopfitness=optim.params$stop.fitness))
#       list(eval=res$value, par=res$par)
#   })
# }
#
#
# OptimizeNM <- function(goal, n, optim.params){
#   MultiOptim(optim.params$retries, function(){
#     ctrl <- list(maxit=optim.params$maxit)
#     res <- optim(runif(n,min=optim.params$par.min,max=optim.params$par.max),goal, control=ctrl)
#
#     list(eval=res$value, par=res$par)
#   })
# }
#
#
# OptimizeBFGS <- function(goal, n, optim.params){
#   MultiOptim(optim.params$retries, function(){
#     ctrl <- list(maxit=optim.params$maxit)
#     res <- optim(runif(n,min=optim.params$par.min,max=optim.params$par.max),goal, control=ctrl, method="BFGS")
#
#     list(eval=res$value, par=res$par)
#   })
# }
#
#
# OptimizeDE <- function(goal, n, optim.params){
#   MultiOptim(optim.params$retries, function(){
#     ctrl <-suppressWarnings( DEoptim::DEoptim.control(itermax=optim.params$maxit, VTR=optim.params$stop.fitness, trace=FALSE, F=optim.params$F, CR=optim.params$CR, strategy= optim.params$strategy))
#     res <- suppressWarnings(DEoptim::DEoptim(goal,lower=rep(optim.params$par.min,n), upper=rep(optim.params$par.max,n), ctrl))
#
#     list(eval=res$optim$bestval, par=res$optim$bestmem)
#   })
# }
#
# OptimizePSO <- function(goal, n, optim.params){
#   MultiOptim(optim.params$retries, function(){
#     control <- list(maxit=optim.params$maxit, trace = FALSE, abstol = optim.params$stop.fitness)
#     out <- pso::psoptim(rep(NA, n), goal, lower=rep(optim.params$par.min,n),upper=rep(optim.params$par.max,n), control=control)
#
#     list(eval=out$value, par=out$par)
#   })
# }
#
# # Scalarization functions ##############################
# ScalFunctions <- function(filter = NULL){
#   # Dot product scalarization
#   DotPr <- function(x,d){ dotprScal(x,d)}
#   expr(DotPr) <- 'dot.pr'
#
#   # Radial scalarization
#   Radial <- function(x,d){radialScal(x,d)}
#   expr(Radial) <- 'radial'
#
#   # Radial scalarization - euclidean distance
#   RootRadial <- function(x,d){rootRadialScal(x,d)}
#   expr(RootRadial) <- 'root.radial'
#
#   res <- list(DotPr, Radial, RootRadial)
#   if(!is.null(filter)){
#     res <- Filter(function(v) any(expr(v)==filter), res)
#   }
#   if(length(res)==0){
#     stop('Invalid scal list given : ', paste(filter,sep=', '))
#   }
#
#   res
# }
#
# ScalParamDim <- function(scal, x){
#   switch(expr(scal),
#          dot.pr = ncol(x)+1,
#          radial = ncol(x)+1,
#          root.radial = ncol(x)+1,
#          stop('Unknown scal: ',expr(scal)))
# }
#
#
#
# # Stop criteria -----------------------------------------------------------
#
#
# IterationSC <- function(max.iterations){
#   list(
#     eval = function(ctx){
#       ctx$continue <- ctx$iteration < ctx$sc$max.iterations
#       ctx
#     },
#     max.iterations = max.iterations
#   )
# }
#
#
# ValidationSC <- function(x,y, max.failures, approximate.timeout = Inf){
#   list(
#     eval = function(ctx){
#       yh <- predict(ctx$m,ctx$sc$x)
#       val <- sqrt(mean((yh-ctx$sc$y)^2))
#       ctx$log(' Validation RMSE: ',val)
#
#
#       if(val >= ctx$sc$last.eval*0.999){
#         ctx$sc$failure.counter <- ctx$sc$failure.counter + 1
#         if(ctx$sc$failure.counter > ctx$sc$max.failures){
#           ctx$log(' Max failed iterations reached')
#
#           ctx$continue <- FALSE
#           ctx$m <- ctx$sc$best.model
#         }
#       }else{
#         ctx$sc$last.eval <- val
#         ctx$sc$best.model <- ctx$m
#         ctx$sc$failure.counter <- 0
#       }
#
#       if(as.numeric(lubridate::now() - ctx$train.start) >= approximate.timeout){
#         ctx$log(' Training timeout')
#         ctx$continue <- FALSE
#       }
#
#       ctx
#     },
#     x = x,
#     y = y,
#     max.failures = max.failures,
#     last.eval = Inf,
#     best.model = list()
#   )
# }
#
# # loadModule("evaluator",what=T)
#
# # IBHM method -------------------------------------------------------------
#
# TrainIBHM <- function(x,y, config = ConfigureIBHM()  ){
#   if(config$jit){
#     require(compiler)
#     setCompilerOptions(suppressAll = TRUE)
#     enableJIT(3)
#   }
#
#   ctx <- CreateContext(x,y,config)
#   ctx$log(str(ctx))
#
#   ctx$m$par$w0 <- mean(y)
#   ctx$yr <- ctx$y-ctx$m$par$w0
#
#   while(ctx$continue){
#     ctx <- ctx$sc$eval( DoIteration(ctx) )
#   }
#
#   ctx$log('Iterations finished RMSE: ', CalcRMSE(ctx))
#   ctx$log('Final parameter estimation ')
#   ctx$m$train$pre.final.model <- ctx$m
#   ctx <- config$final.estimation(ctx, x = ctx$x.final, y = ctx$y.final, maxit = config$final.estimation.maxit)
#
#   ctx$log('\nFinal RMSE: ',CalcRMSE(ctx))
#
#   return(ctx$m)
# }
#
#
# DoIteration <- function(prevCtx){
#   ctx <- prevCtx
#   ctx$log('\nIteration ',ctx$iteration, ' RMSE: ', CalcRMSE(ctx))
#   ctx$iteration <- ctx$iteration + 1
#
#   ctx <- FindScal(ctx)
#   if(ctx$continue){
#     ctx <- FindActiv(ctx)
#     ctx$m$par$w <- append(ctx$m$par$w,1)
#     ctx <- OptimizeWeights(ctx)
#     UpdateResidual(ctx)
#   }else{
#     ctx$log('Stopping')
#     prevCtx$continue <- F
#     prevCtx
#   }
# }
#
# CalcRMSE <- function(ctx){
#   sqrt(mean((ctx$y-predict.IBHM(ctx$m,ctx$x))^2))
# }
#
#
#
# # Method runtime context --------------------------------------------------
#
# CreateContext <- function(x,y, params){
#   x <- as.matrix(x)
#   y <- as.matrix(y)
#   stopifnot(nrow(x)==nrow(y), nrow(x)>10, is.null(params$final.estimation.x) == is.null(params$final.estimation.y))
#
#
#   if(params$verbose){
#     log <- function(...) cat(...,'\n')
#   }else{
#     log <- function(...) {}
#   }
#
#   x.m <- max(abs(x))
#   params$scal$optim.params$par.min <- min(-100,-x.m)
#   params$scal$optim.params$par.max <- max(100, x.m)
#
#   params$activ$optim.params$par.min <- min(-100,-x.m)
#   params$activ$optim.params$par.max <- max(100,x.m)
#
#   sc.mask <- sample(nrow(x),ceiling(nrow(x)*params$sc.frac))
#
#   append(params,
#          list( x = as.matrix(x[-sc.mask,]),
#                y = as.matrix(y[-sc.mask]),
#                yr = y,
#                x.final = x,
#                y.final = y,
#                m=CreateEmptyModel(as.matrix(x[-sc.mask,]),as.matrix(y[-sc.mask]), train.config = params$unparsed.config, filtering.function = params$wf),
#                scal.y = list(),
#                continue=TRUE,
#                iteration=0,
#                log = log,
#                sc = ValidationSC(as.matrix(x[sc.mask,]),as.matrix(y[sc.mask]), params$sc.max.failures, params$sc.approximate.timeout),
#                train.start = lubridate::now()
#          )
#   )
# }
#
#
# # IBHM model structure ----------------------------------------------------
# CreateEmptyModel <- function(x=NULL,y=NULL, train.config = NULL, filtering.function){
#   m <- list(par = list(
#                 w0=0,
#                 w=vector(),
#                 d=list(),
#                 a=vector(),
#                 b=vector()),
#             scals=list(),
#             activs=list(),
#             train=list(x=x,y=y, filtering.function = filtering.function, w.pars = vector(), config = train.config))
#   class(m) <- 'IBHM'
#   return(m)
# }
#
#
#
# # Standard S3 methods -----------------------------------------------------
# predict.IBHM <- function(object,x = NULL,...){
#   m<-object
#   if(is.null(x)){
#     x <- m$train$x
#   }else{
#     x <- as.matrix(x)
#   }
#
#   d <- dim(x)
#   len <- length(m$scals)
#   y <- matrix(m$par$w0,d[[1]],1)
#
#
#   if(len > 0){
#     for(i in 1:len){
#       act <- m$activs[[i]]
#       a <- m$par$a[[i]]
#       b <- m$par$b[[i]]
#
#       scal <- m$scals[[i]]
#       d <- m$par$d[[i]]
#
#       w <- m$par$w[[i]]
#
#       y <- y + w * act( a*scal(x,d)+b)
#     }
#   }
#
#   y
# }
#
#
# length.IBHM <- function(x){
#   length(x$par$w)
# }
#
# ToString <- function(m){
#   f <- function(v){
#     sprintf('%.2e',v)
#   }
#
#   str <- f(m$par$w0)
#   len <- length(m$par$w)
#
#   if(len>0){
#     for(i in 1:length(m)){
#       w <- m$par$w[[i]]
#       if(w>0) s<-'+' else s<-''
#
#       act <- expr(m$activs[[i]])
#       a <- f(m$par$a[[i]])
#       b <- f(m$par$b[[i]])
#
#       scal <- expr(m$scals[[i]])
#       d <- paste(f(m$par$d[[i]]),collapse=' ')
#
#
#       str <- paste(str,s,f(w),act,' (',a,'*',scal,'(x,[',d,'])','+',b,') ')
#     }
#   }
#
#   str
#
# }
#
# summary.IBHM <- function(object,...){
#   y <- object$train$y
#   x <- object$train$x
#
#   se <- c((y - predict(object))^2)
#
#   res <- list(model = ToString(object),
#               model.size = length(object),
#               TrainSize = nrow(y),
#               TrainDim = ncol(x),
#               mse = mean(se),
#               sd.se = sd(se),
#               rmse = sqrt(mean(se)),
#               cor=cor(predict(object),y))
#   class(res) <- 'summary.IBHM'
#   res
# }
#
# print.summary.IBHM <- function(x,...){
#   cat('Model equation: ', x$model,'\n',
#       'Model size: ',x$model.size,'\n\n',
#       'Train set dim: ', x$TrainDim, ' Train set size: ', x$TrainSize,'\n',
#       'MSE:  ', x$mse, ' Std. dev:', x$se.sd,'\n',
#       'RMSE: ', x$rmse, '\n',
#       'Pearson correlation coefficient: ', x$cor,'\n')
# }
#
#
#
# regularize <- function(params){
#   #max(sqrt(sum(params^2))-1,0)
#   sum(abs(params))
# }
#
# # Finding a scalarization function ----------------------------------------
#
# FindScal <- function(ctx){
#   within(ctx,{
#     log(' Finding next scal')
#
#     best <- list(scal=NULL, eval = Inf, d = NULL)
#
#     for( s in  scal$candidates){
#       candidate <- OptimizeScal(s,ctx)
#       log('  ',expr(s), ' eval : ', candidate$eval)
#
#       if(candidate$eval < best$eval){
#         best <- candidate
#       }
#     }
#     if(is.finite(best$eval)){
#       log(' Best scal: ',expr(best$scal), ' d: ', best$d,' w.par:',best$w.par)
#
#       m$scals[[iteration]]  <- best$scal
#       m$par$d[[iteration]] <- best$d
#       m$train$w.pars[[iteration]] <- best$w.par
#
#       zk <- best$scal(x, best$d)
#       w <- wf(zk, best$w.par)
#
#     }else{
#       log(' Could not find next scal - stopping.')
#       continue <- F
#     }
#   })
# }
#
#
# OptimizeScal <- function(scal, ctx){
#
#   goal <- function(par, lambda){
#     zCandidate <- scal(ctx$x, par)
#     w <- ctx$wf(zCandidate, ctx$weighting.par)
#
#     cor.y <- weightedR(ctx$yr, zCandidate,w)
#
#     if(is.nan(cor.y)){
#       cor.y<-0
#     }
#
#     return(1 - abs(cor.y)  + lambda*regularize(par))
#   }
#
#   n <- ScalParamDim(scal, ctx$x)
#   res <- ctx$scal$optim(function(par) goal(par,ctx$scal$lambda),n,ctx$scal$optim.params)
#
#   list(scal = scal, eval = res$eval, w.par=ctx$weighting.par, d=res$par)
# }
#
#
# # Finding an activation function #####################
#
#
# FindActiv <- function(ctx){
#   within(ctx, {
#     log(' Finding next activ')
#
#     zk <- RunLastScal(ctx)
#     best <- NULL
#     best.eval <- Inf
#
#     for(i in 1:length(activ$candidates)){
#       cand <- OptimizeActiv(activ$candidates[[i]], zk, ctx)
#
#       if(cand$eval < best.eval){
#         best <- cand
#         best.eval <- cand$eval
#       }
#     }
#
#     m$activs[[iteration]] <- best$activ
#     m$par$a[[iteration]]  <- best$a
#     m$par$b[[iteration]]  <- best$b
#
#     log(' Best activ: ', expr(best$activ), ' a:',best$a, ' b:',best$b)
#   })
# }
#
#
# OptimizeActiv <- function(activ, zk, ctx){
#   goal <- function(par, lambda){
#     a <- par[[1]]
#     b <- par[[2]]
#     vk <- activ(a*zk+b)
#     val<-weightedR(ctx$yr, vk, ctx$w)
#
#     if(is.nan(val) || !is.finite(val)){
#       val<-0
#     }
#
#     eval <- 1-abs(val) + lambda*regularize(par)
#
#     if(is.nan(eval)){
#       eval <- Inf
#     }
#     return(eval)
#   }
#
#   res <- ctx$activ$optim(function(par) goal(par, ctx$scal$lambda), 2, ctx$activ$optim.params)
#
#   ctx$log('  ',expr(activ),' eval:',res$eval)
#   list(activ=activ, eval=res$eval,  a=res$par[[1]], b=res$par[[2]], v = activ(res$par[[1]]*zk+res$par[[2]]))
# }
#
# RunLastScal <-function(ctx){
#   scal <- ctx$m$scals[[ctx$iteration]]
#   d <- ctx$m$par$d[[ctx$iteration]]
#
#   scal(ctx$x,d)
# }
#
# RunLastActiv <-function(ctx){
#   activ<- ctx$m$activs[[ctx$iteration]]
#   a <- ctx$m$par$a[[ctx$iteration]]
#   b <- ctx$m$par$b[[ctx$iteration]]
#
#   activ(a*RunLastScal(ctx)+b)
# }
#
#
#
# # Estimating regression weights -------------------------------------------
#
# OptimizeWeights <- function(ctx){
#   vk <- RunLastActiv(ctx)
#   vk_bar <- vk - weightedMean(vk,ctx$w)
#
#   yr_bar <- ctx$yr - weightedMean(ctx$yr,ctx$w)
#
#   goal <- function(w){
#     return(weightedMean((yr_bar - w*vk_bar)^2 ,ctx$w) )
#   }
#
#   res <- optim(1,goal, method="BFGS")
#
#   ctx$m$par$w[[ctx$iteration]] <- res$par
#   ctx$m$par$w0<- ctx$m$par$w0 - res$par*mean(vk)
#
#   ctx$log(' w: ', res$par,' w0: ', ctx$m$par$w0, ' dw0: ', -mean(vk))
#
#   ctx
# }
#
# OptimizeAllWeights <- function(ctx, x = ctx$x, y = ctx$y, maxit=100){
#   within(ctx,{
#     log(' Optimizing all weights')
#
#     goal <- function(par){
#       m$par$w0 <- par[[1]]
#       m$par$w <- par[2:length(par)]
#
#       return(mean((y - predict.IBHM(m, x))^2))
#     }
#
#     res <- optim(c(m$par$w0, m$par$w),goal, method="BFGS", control=list(maxit=maxit, trace=F))
#
#     m$par$w0 <- res$par[[1]]
#     m$par$w <- res$par[2:length(res$par)]
#   })
# }
#
# OptimizeAllParams <- function(ctx,x = ctx$x, y = ctx$y, maxit=200){
#   within(ctx,{
#     log(' Optimizing all parameters')
#
#     goal <- function(par){
#       m$par <- relist(par, m$par)
#       return(mean((y - predict.IBHM(m,x))^2))
#     }
#
#     res <- optim(unlist(m$par),goal, method="BFGS", control=list(maxit=maxit, trace=F))
#     m$par <- relist(res$par, m$par)
#   })
# }
#
# UpdateResidual <- function(ctx){
#   ctx$yr <- ctx$y - predict.IBHM(ctx$m, ctx$x)
#   ctx
# }
#
# # Utilities ############################################
#
# # Attribute used while printing the model expression
# expr <- function(obj){
#   return(attr(obj,'expr'))
# }
#
# 'expr<-' <- function(obj, value){
#   attr(obj,'expr') <- value
#   return(obj)
# }
#
# latex.tab.content <- function(d){
#   nrows <- dim(d)[[1]]
#   ncols <- dim(d)[[2]]
#
#
#   printRow <- function(rd){
#     if(is.na(rd[[1]])){
#       row <- '';
#     }else{
#       row <- paste(rd[[1]])
#     }
#     for(j in 2:ncols){
#       if(is.na(rd[[j]])){
#         row <- paste(row,'&')
#       }else if(is.numeric(rd[[j]])){
#         row <- paste(row,sprintf(' & %.2e',rd[[j]]))
#       }else{
#
#         row <- paste(row,sprintf(' & %s',rd[[j]]))
#       }
#     }
#     cat(paste(row,' \\\\\n'))
#   }
#
#   printRow(colnames(d))
#   for(i in 1:nrows){
#     cat(rownames(d)[[i]],' & ')
#     printRow(d[i,])
#   }
# }
#
#
#
#
# group.by.d_start <- function(d){
#   d$d_start <- round(d$d_start,2)
#
#   evals <- levels(d$eval_name)
#
#   df <- data.frame(d_start <- unique(d$d_start))
#
#   nrows <- dim(df)[[1]]
#   for(ev in evals){
#     col <- matrix(0,nrows,1)
#     colnames(col) <- ev
#     for(row in 1:nrows){
#       mask <- (abs(d$d_start - df$d_start[[row]])<0.01 & d$eval_name == ev)
#       col[[row]] <- mean(d$d_stop[mask])
#     }
#
#     df <- cbind(df,col)
#   }
#
#   return(df)
# }
#
#
# group.by.samplesPerDim <- function(d){
#
#   evals <- levels(d$eval_name)
#
#   df <- data.frame(SamplesPerDim = unique(d$SamplesPerDim))
#
#   nrows <- dim(df)[[1]]
#
#   for(ev in evals){
#     col <- matrix(0,nrows,1)
#     sdCol <- matrix(0,nrows,1)
#     colnames(col) <- ev
#     colnames(sdCol) <- paste(ev,'_sd')
#     for(row in 1:nrows){
#       mask <- (d$SamplesPerDim == df$SamplesPerDim[[row]] & d$eval_name == ev)
#       col[[row]] <- mean(d$d_stop[mask])
#       sdCol[[row]] <- sd(d$d_stop[mask])
#     }
#
#     df <- cbind(df,col,sdCol)
#   }
#
#   return(df)
# }
#
# radialScal <- function(x, dd) {
#     .Call('IBHM_radialScal', PACKAGE = 'IBHM', x, dd)
# }
#
# rootRadialScal <- function(x, dd) {
#     .Call('IBHM_rootRadialScal', PACKAGE = 'IBHM', x, dd)
# }
#
# dotprScal <- function(x, dd) {
#     .Call('IBHM_dotprScal', PACKAGE = 'IBHM', x, dd)
# }
#
# weightedMean <- function(x, w) {
#     .Call('IBHM_weightedMean', PACKAGE = 'IBHM', x, w)
# }
#
# weightedVar <- function(x, w) {
#     .Call('IBHM_weightedVar', PACKAGE = 'IBHM', x, w)
# }
#
# weightedCov <- function(x, z, w) {
#     .Call('IBHM_weightedCov', PACKAGE = 'IBHM', x, z, w)
# }
#
# weightedR <- function(y1, y2, w) {
#     .Call('IBHM_weightedR', PACKAGE = 'IBHM', y1, y2, w)
# }
#
# weightedKendall <- function(y1Ranked, y2Ranked, w) {
#     .Call('IBHM_weightedKendall', PACKAGE = 'IBHM', y1Ranked, y2Ranked, w)
# }
#
# tiedRanks <- function(x) {
#     .Call('IBHM_tiedRanks', PACKAGE = 'IBHM', x)
# }
#
# weightedRho <- function(y1, y2, w) {
#     .Call('IBHM_weightedRho', PACKAGE = 'IBHM', y1, y2, w)
# }
#             '''

            '''
            ibhm2Learner <- function(){
  sqr <- function(y,...){
     w <- y*0 + 0.0001
     w[abs(y)<0.1] <- 1
     w
  }
  require(IBHM)
  makeLearner("regr.IBHM", id = "IBHM_2") %>% 
      setHyperPars(final.estimation.maxit = 20
                   , stop.criterion.max.failures = 1
                   , verbose = FALSE 
                   , scal.lambda = 1e-9
                   , optim.retries = 3
                   , weighting.function = sqr)
}

TrainIBHM <- function(x,y, config = ConfigureIBHM()  ){
  if(config$jit){    
    require(compiler)
    setCompilerOptions(suppressAll = TRUE)
    enableJIT(3)
  }  
  
  ctx <- CreateContext(x,y,config)
  ctx$log(str(ctx))
  
  ctx$m$par$w0 <- mean(y)  
  ctx$yr <- ctx$y-ctx$m$par$w0
  
  while(ctx$continue){      
    ctx <- ctx$sc$eval( DoIteration(ctx) )    
  }      
  
  ctx$log('Iterations finished RMSE: ', CalcRMSE(ctx))
  ctx$log('Final parameter estimation ')
  ctx$m$train$pre.final.model <- ctx$m
  ctx <- config$final.estimation(ctx, x = ctx$x.final, y = ctx$y.final, maxit = config$final.estimation.maxit)
  
  ctx$log('\nFinal RMSE: ',CalcRMSE(ctx))
  
  return(ctx$m)
}


predict.IBHM <- function(object,x = NULL,...){  
  m<-object
  if(is.null(x)){
    x <- m$train$x
  }else{
    x <- as.matrix(x)  
  }  
  
  d <- dim(x)
  len <- length(m$scals)  
  y <- matrix(m$par$w0,d[[1]],1)
  
  
  if(len > 0){
    for(i in 1:len){
      act <- m$activs[[i]]
      a <- m$par$a[[i]]
      b <- m$par$b[[i]]
            
      scal <- m$scals[[i]]
      d <- m$par$d[[i]]
      
      w <- m$par$w[[i]]                       
      
      y <- y + w * act( a*scal(x,d)+b)
    }
  }
  
  y
}
            '''
        )


    def trainAndCompileModel(self):
        self.rIBHMTrainer = robjects.globalenv["TrainIBHM"]
        model = robjects.globalenv["predict.IBHM"]

        # training, validation = self.generateTrainingData()
        training= self.generateTrainingData()
        evaluatedTraining = self.evaluateSet(training)
        # evaluatedValidation = self.evaluateSet(validation)

        evaluatedTraining = [item for sublist in evaluatedTraining for item in sublist]
        self.trainingEvaluatedPairSet = [ (item, evaluatedTraining[i]) for i,item in enumerate(training)]

        training = robjects.FloatVector(training)
        evaluatedTraining = robjects.FloatVector(evaluatedTraining)
        self.rIBHMTrainer(training, evaluatedTraining)
        return model

    def generateTrainingData(self):
        # data = np.random.uniform(minValue, maxValue, (params.trainingDataSize, dimensions))
        lhsData = lhs(self.dEParams.dimensions, samples=self.dESVRParams.trainingDataSize)
        data = np.array(lhsData) * 200
        data = data - 100
        # cut = np.int32(self.dESVRParams.trainingDataSize * 0.8)
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
        specimen_val = self.model(np.array(specimen).reshape(1, self.dEParams.dimensions))
        trialVector_val = self.model(np.array(trialVector).reshape(1, self.dEParams.dimensions))

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
                evaluatedPopulation = self.evaluateSet(population)
                fes += 20

                evaluatedPopulation = [item for sublist in evaluatedPopulation for item in sublist]

                self.rIBHMTrainer(population, evaluatedPopulation)

            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")

            MSE = mean_squared_error(originalValuesForMSE, modelValuesForMSE)
            print("MSE: " + str(MSE))
            generation += 1
        self.dEParams.seed += 1
        return population

    def run(self):
        random.seed(self.dEParams.seed)
        np.random.seed(self.dEParams.seed)



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
                selectedPopulation = population.choose(20)
                evaluatedPopulation = self.evaluateSet(selectedPopulation)
                fes += 20

                evaluatedPopulation = [item for sublist in evaluatedPopulation for item in sublist]
                print(evaluatedPopulation)

                self.updateTrainingSet(selectedPopulation, evaluatedPopulation)
                training = [ x for (x, _) in self.trainingEvaluatedPairSet]
                evaluatedTraining = [ x for (_, x) in self.trainingEvaluatedPairSet]

                self.rIBHMTrainer(training, evaluatedTraining)

            if (self.dEParams.debug):
                print("Generation: " + str(fes / self.dEParams.populationSize))
                randomSpecimen = population[np.random.randint(population.shape[0], size=1), :][0]
                print("Sample from population: " + str(randomSpecimen))
                print("Value evaluated with model: " + str(self.model.predict(np.array(randomSpecimen).reshape(1, self.dEParams.dimensions))) + "\n")

            generation += 1
        self.dEParams.seed += 1
        return population


    def updateTrainingSet(self, newTrainingPoints, evaluatedNewTrainingPoints):
        sortedTrainingSet = sorted(self.trainingEvaluatedPairSet, key=lambda x: x[1])[:self.dESVRParams.trainingDataSize-20]

        for i,item in enumerate(newTrainingPoints):
            sortedTrainingSet.append((item, evaluatedNewTrainingPoints[i]))

        self.trainingEvaluatedPairSet = sortedTrainingSet






