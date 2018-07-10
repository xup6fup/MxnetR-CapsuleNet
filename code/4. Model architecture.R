library(mxnet)

data <- mx.symbol.Variable(name = 'data')
label <- mx.symbol.Variable(name = 'label')

# Capsule function

My_activation <- function (indata, squash_axis, name = '', eps = 1e-08) {
  
  squared_indata <- mx.symbol.square(data = indata, name = paste0('square_', name))
  
  sum_squared_indata <- mx.symbol.sum(data = squared_indata, axis = squash_axis,
                                      keepdims = TRUE, name = paste0('sum_square_', name))
  
  sqrt_sum_squared_indata <- mx.symbol.sqrt(data = sum_squared_indata, name = paste0('sqrt_sum_squared_', name))
  
  
  scale <- sum_squared_indata / (1 + sum_squared_indata) / sqrt_sum_squared_indata
  
  squashed_net <- mx.symbol.broadcast_mul(lhs = scale, rhs = indata, name = paste0('squashed_net_', name))
  
  return(squashed_net)
  
}

Primary_caps <- function (indata, dim_vector, n_channels, kernel, strides, name = 'P_caps') {
  
  out <- mx.symbol.Convolution(data = indata,
                               kernel = kernel, num_filter = dim_vector * n_channels, stride = strides,
                               name = paste0('conv_', name))
  
  out <- mx.symbol.reshape(data = out, shape = c(-1, dim_vector, 1, 0), name = paste0('reshape_', name))
  
  out <- My_activation(indata = out, squash_axis = 3, name = name, eps = 1e-08)
  
  return(out)
  
}

Get_Agree <- function (indata, axis, squared = FALSE, eps = 1e-3, name = '') {
  
  if (squared) {
    
    m <- mx.symbol.mean(data = indata, axis = axis, keepdims = TRUE, name = paste0('mean1_', name))
    
    res <- mx.symbol.broadcast_minus(lhs = indata, rhs = m, name = paste0('residual_', name))
    
    res2 <- mx.symbol.square(data = res, name = paste0('square_residual_', name))
    
    sqrt_res2 <- mx.symbol.sqrt(data = res2 + eps, name = paste0('sqrt_square_residual_', name))
    
    weight_indata <- mx.symbol.broadcast_div(lhs = indata, rhs = sqrt_res2, name = paste0('weighted_', name))
    
    agree <- mx.symbol.mean(data = weight_indata, axis = axis, keepdims = TRUE, name = paste0('agree_', name))
    
  } else {
    
    agree <- mx.symbol.mean(data = indata, axis = axis, keepdims = TRUE, name = paste0('agree_', name))
    
  }
  
  
  return(agree)
  
}

Digit_caps <- function (indata, kernel, strides, num_filter, num_output, name = 'D_caps') {
  
  out <- mx.symbol.Convolution(data = indata, no.bias = TRUE,
                               kernel = kernel, num_filter = num_filter * num_output, stride = strides,
                               name = paste0('conv_', name))
  
  out <- mx.symbol.reshape(data = out, shape = c(-1, num_filter, num_output, 0), name = paste0('reshape_', name))
  
  out <- Get_Agree(indata = out, axis = 3, squared = TRUE, eps = 1e-3, name = name)
  
  return(out)
  
}

CapsNet <- function (indata, num_filter1 = 128, dim_vector = 8, n_channels = 16, num_filter2 = 16, num_output = 10) {
  
  # first conv
  
  conv1 <- mx.symbol.Convolution(data = indata, kernel = c(9, 9), num_filter = num_filter1, name = 'conv1')
  relu1 <- mx.symbol.Activation(data = conv1, act_type = "relu", name = 'relu1')
  
  # first capsule
  
  P_caps <- Primary_caps(indata = relu1, dim_vector = dim_vector, n_channels = n_channels,
                         kernel = c(9, 9), strides = c(2, 2), name = 'P_caps')
  
  # Routing by Agreement
  
  D_caps <- Digit_caps(indata = P_caps, kernel = c(1, dim_vector), strides = c(1, 1),
                       num_filter = num_filter2, num_output = num_output, name = 'D_caps')
  
  return(D_caps)
  
}

# LeNet function

pred_out <- CapsNet(indata = data)

square_D_caps <- mx.symbol.square(data = pred_out, name = 'square_D_caps')

sum_D_caps <- mx.symbol.sum(data = square_D_caps, axis = 2:3, name = 'sum_D_caps')

softmax <- mx.symbol.softmax(data = sum_D_caps, axis = 1, name = 'softmax')

m_log_loss <- 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(softmax + 1e-8), label))

final_agree_list <- mx.symbol.SliceChannel(data = pred_out, num.outputs = 10, axis = 1,
                                           squeeze.axis = FALSE, name = 'final_agree_list')

fig_list <- list()

for (i in 1:10) {
  
  deconv1 <- mx.symbol.Convolution(data = final_agree_list[[i]],
                                   kernel = c(1, 16), num_filter = 128, stride = c(1, 1),
                                   name = paste0('deconv1_', i))
  
  derelu1 <- mx.symbol.Activation(data = deconv1, act_type = "relu", name = paste0('derelu1_', i))
  
  deconv2 <- mx.symbol.Deconvolution(data = derelu1, 
                                     kernel = c(1, 1), num_filter = 256, stride = c(1, 1),
                                     name = paste0('deconv2_', i))
  
  derelu2 <- mx.symbol.Activation(data = deconv2, act_type = "relu", name = paste0('derelu2_', i))
  
  fig_list[[i]] <- mx.symbol.Deconvolution(data = derelu2, 
                                           kernel = c(28, 28), num_filter = 1, stride = c(1, 1),
                                           name = paste0('fig_list_', i))
  
}

final_fig <- mx.symbol.concat(data = fig_list, num.args = 10, dim = 1, name = 'final_fig')

reshape_label <- mx.symbol.reshape(data = label, shape = c(1, 1, 10, 0), name = 'reshape_agree')

recovery_img <- mx.symbol.sum(mx.symbol.broadcast_mul(final_fig, reshape_label), keepdims = TRUE, axis = 1)

mse_loss <- mx.symbol.mean(mx.symbol.square(recovery_img - data))

final_loss <- mx.symbol.MakeLoss(m_log_loss + mse_loss * 5e-4, name = 'final_loss')

#mx.symbol.infer.shape(recovery_img, data = c(28, 28, 1, 7), label = c(10, 7))$out.shapes
