library(mxnet)

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "data.csv", "data.shape", "batch.size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, data.csv, data.shape, batch.size){
                                    csv_iter <- mx.io.CSVIter(data.csv = data.csv, data.shape = data.shape, batch.size = batch.size)
                                    .self$iter <- csv_iter
                                    .self
                                  },
                                  value = function(){
                                    val <- as.array(.self$iter$value()$data)
                                    val.x <- val[-1,]
                                    val.y <- t(model.matrix(~ -1 + factor(val[1,], levels = 0:9)))
                                    val.y <- array(val.y, dim = c(10, ncol(val.x)))
                                    dim(val.x) <- c(28, 28, 1, ncol(val.x))
                                    val.x <- mx.nd.array(val.x)
                                    val.y <- mx.nd.array(val.y)
                                    list(data=val.x, label=val.y)
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

my_iter = my_iterator_func(iter = NULL,  data.csv = 'data/train_data.csv', data.shape = 785, batch.size = 20)

#my_iter$reset()
#my_iter$iter.next()
#my_value = my_iter$value()
#library(OpenImageR)
#imageShow(matrix(as.numeric(as.array(my_value$data)[,,,20]), nrow = 28, byrow = TRUE))