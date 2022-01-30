# Scalarization functions ##############################
ScalFunctions <- function(filter = NULL){
  # Dot product scalarization
  DotPr <- function(x,d){ dotprScal(x,d)}
  expr(DotPr) <- 'dot.pr'

  # Radial scalarization
  Radial <- function(x,d){radialScal(x,d)}
  expr(Radial) <- 'radial'

  # Radial scalarization - euclidean distance
  RootRadial <- function(x,d){rootRadialScal(x,d)}
  expr(RootRadial) <- 'root.radial'
  
  res <- list(DotPr, Radial, RootRadial)  
  if(!is.null(filter)){
    res <- Filter(function(v) any(expr(v)==filter), res)
  }
  if(length(res)==0){
    stop('Invalid scal list given : ', paste(filter,sep=', '))
  }
  
  res
}

ScalParamDim <- function(scal, x){
  switch(expr(scal), 
         dot.pr = ncol(x)+1,
         radial = ncol(x)+1,
         root.radial = ncol(x)+1,
         stop('Unknown scal: ',expr(scal)))
}