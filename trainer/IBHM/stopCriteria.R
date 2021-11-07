
# Stop criteria -----------------------------------------------------------


IterationSC <- function(max.iterations){
  list( 
    eval = function(ctx){
      ctx$continue <- ctx$iteration < ctx$sc$max.iterations
      ctx
    },        
    max.iterations = max.iterations
  )    
}


ValidationSC <- function(x,y, max.failures, approximate.timeout = Inf){
  list( 
    eval = function(ctx){              
      yh <- predict(ctx$m,ctx$sc$x)
      val <- sqrt(mean((yh-ctx$sc$y)^2))
      ctx$log(' Validation RMSE: ',val)
            
      
      if(val >= ctx$sc$last.eval*0.999){
        ctx$sc$failure.counter <- ctx$sc$failure.counter + 1
        if(ctx$sc$failure.counter > ctx$sc$max.failures){
          ctx$log(' Max failed iterations reached')
          
          ctx$continue <- FALSE
          ctx$m <- ctx$sc$best.model
        }
      }else{
        ctx$sc$last.eval <- val
        ctx$sc$best.model <- ctx$m
        ctx$sc$failure.counter <- 0
      }
      
      if(as.numeric(lubridate::now() - ctx$train.start) >= approximate.timeout){
        ctx$log(' Training timeout')
        ctx$continue <- FALSE
      }
      
      ctx      
    },        
    x = x,
    y = y,    
    max.failures = max.failures,
    last.eval = Inf,
    best.model = list()
  )
}