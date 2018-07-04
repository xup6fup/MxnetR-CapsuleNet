library(mxnet)
library(magrittr)
library(data.table)

Capsnet_model <- mx.model.load('CapsNet', 0)

all_layers = Capsnet_model$symbol$get.internals()
softmax_output = which(all_layers$outputs == 'softmax_output') %>% all_layers$get.output()

Capsnet_model$symbol <- softmax_output
Capsnet_model$arg.params <- Capsnet_model$arg.params[names(Capsnet_model$arg.params) %in% names(mx.symbol.infer.shape(softmax_output, data = c(28, 28, 1, 7))$arg.shapes)]
Capsnet_model$aux.params <- Capsnet_model$aux.params[names(Capsnet_model$aux.params) %in% names(mx.symbol.infer.shape(softmax_output, data = c(28, 28, 1, 7))$aux.shapes)]

DAT = fread("data/test_data.csv", data.table = FALSE)
DAT = data.matrix(DAT)

Test.X = t(DAT[,-1]/255)
dim(Test.X) = c(28, 28, 1, ncol(Test.X))
Test.Y = DAT[,1]

predict_Y = predict(Capsnet_model, Test.X, ctx = mx.gpu())
confusion_table = table(max.col(t(predict_Y)), Test.Y)
cat("Testing accuracy rate =", sum(diag(confusion_table))/sum(confusion_table))