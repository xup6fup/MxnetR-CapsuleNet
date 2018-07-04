library(mxnet)
library(data.table)

Capsnet_model <- mx.model.load('CapsNet', 0)

DAT = fread("data/test_data.csv", data.table = FALSE)
DAT = data.matrix(DAT)

Test.X = t(DAT[,-1])
dim(Test.X) = c(28, 28, 1, ncol(Test.X))
Test.Y = DAT[,1]

predict_Y = predict(Capsnet_model, Test.X, ctx = mx.gpu())
confusion_table = table(max.col(t(predict_Y)), Test.Y)
cat("Testing accuracy rate =", sum(diag(confusion_table))/sum(confusion_table))