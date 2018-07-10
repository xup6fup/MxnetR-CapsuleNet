library(mxnet)
library(imager)

digit <- 6

Capsnet_model <- mx.model.load('CapsNet', 0)

vec <- runif(16, -3, 3)

fc1_out <- vec %*% as.array(Capsnet_model$arg.params[[paste0('deconv1_', digit + 1, '_weight')]])[1,,,] + as.array(Capsnet_model$arg.params[[paste0('deconv1_', digit + 1, '_bias')]])
relu1_out <- fc1_out
relu1_out[relu1_out < 0] <- 0

fc2_out <- fc1_out %*% t(as.array(Capsnet_model$arg.params[[paste0('deconv2_', digit + 1, '_weight')]])[1,,,])
relu2_out <- fc2_out
relu2_out[relu2_out < 0] <- 0

for (k in 1:256) {
  if (k == 1) {
    img_out <- relu2_out[,k] * as.array(Capsnet_model$arg.params[[paste0('fig_list_', digit + 1, '_weight')]])[,,,k]
  } else {
    img_out <- img_out + relu2_out[,k] * as.array(Capsnet_model$arg.params[[paste0('fig_list_', digit + 1, '_weight')]])[,,,k]
  }
}

img_out[img_out > 1] <- 1
img_out[img_out < 0] <- 0

par(mar=rep(0,4))

plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
rasterImage(as.cimg(img_out), 0, 0, 1, 1, interpolate=FALSE)