library(mxnet)

my_optimizer = mx.opt.create(name = "adam", learning.rate = 0.001, beta1 = 0.9, beta2 = 0.999,
                             epsilon = 1e-08, wd = 1e-4)

#my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0.9, wd = 1e-4)
