library(mxnet)
library(magrittr)
library(data.table)
library(imager)

Capsnet_model <- mx.model.load('CapsNet', 0)

my_predict <- function (model, X, label = NULL, ctx) {
  
  all_layers = model$symbol$get.internals()
  agree_D_caps_output = which(all_layers$outputs == 'agree_D_caps_output') %>% all_layers$get.output()
  softmax_output = which(all_layers$outputs == 'softmax_output') %>% all_layers$get.output()
  final_fig_output = which(all_layers$outputs == 'final_fig_output') %>% all_layers$get.output()
  
  #needed_arg <- names(mx.symbol.infer.shape(softmax_output, data = c(28, 28, 1, 7))$arg.shapes)
  
  out = mx.symbol.Group(c(softmax_output, agree_D_caps_output, final_fig_output))
  executor = mx.simple.bind(symbol = out, data = dim(X), ctx = ctx)
  
  #mx.exec.update.arg.arrays(executor, model$arg.params[names(model$arg.params) %in% needed_arg], match.name = TRUE)
  mx.exec.update.arg.arrays(executor, model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(executor, model$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(executor, list(data = mx.nd.array(X)), match.name = TRUE)
  mx.exec.forward(executor, is.train = FALSE)
  
  capsule_feature = as.array(executor$ref.outputs$agree_D_caps_output)
  pred_out = as.array(executor$ref.outputs$softmax_output)
  fig_out = as.array(executor$ref.outputs$final_fig_output)
  
  if (is.null(label)) {
    label <- which.max(pred_out) - 1
  }
  
  img = array(fig_out[,,label+1,], dim = dim(X))
  img[img > 1] <- 1
  img[img < 0] <- 0
  
  return(list(capsule_feature = capsule_feature, pred_out = pred_out, fig_out = fig_out, img = img))
  
}

DAT = fread("data/test_data.csv", data.table = FALSE)
DAT = data.matrix(DAT)

Test.X = t(DAT[,-1]/255)
dim(Test.X) = c(28, 28, 1, ncol(Test.X))
Test.Y = DAT[,1]


par(mar=rep(0,4), mfcol = c(4, 5))

for (i in 1:10) {
  
  pos <- sample(which(Test.Y == i - 1), 1)
  
  Input <- Test.X[,,,pos]
  dim(Input) <- c(28, 28, 1, 1)
  
  predict_Y = my_predict(model = Capsnet_model, X = Input, ctx = mx.gpu())
  
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
  rasterImage(as.cimg(Input[,,,1]), 0, 0, 1, 1, interpolate=FALSE)
  
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
  rasterImage(as.cimg(predict_Y$img[,,,1]), 0, 0, 1, 1, interpolate=FALSE)
  
}