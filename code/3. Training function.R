library(mxnet)

my.model.FeedForward.create = function (Iterator, ctx = mx.cpu(), save.grad = FALSE,
                                        loss_symbol, pred_symbol,
                                        Optimizer, num_round = 30) {
  
  require(abind)
  
  #0. Check data shape
  Iterator$reset()
  Iterator$iter.next()
  my_values <- Iterator$value()
  input_shape <- lapply(my_values, dim)
  batch_size <- tail(input_shape[[1]], 1)
  
  #1. Build an executor to train model
  exec_list = list(symbol = loss_symbol, ctx = ctx, grad.req = "write")
  exec_list = append(exec_list, input_shape)
  my_executor = do.call(mx.simple.bind, exec_list)
  
  #2. Set the initial parameters
  mx.set.seed(0)
  new_arg = mxnet:::mx.model.init.params(symbol = loss_symbol,
                                         input.shape = input_shape,
                                         output.shape = NULL,
                                         initializer = mxnet:::mx.init.uniform(0.01),
                                         ctx = ctx)
  mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)
  
  print(sum(sapply(lapply(new_arg$arg.params, dim), prod)))
  
  #3. Define the updater
  my_updater = mx.opt.get.updater(optimizer = Optimizer, weights = my_executor$ref.arg.arrays)
  
  #4. Forward/Backward
  message('Start training:')
  
  set.seed(0)
  if (save.grad) {epoch_grad = NULL}
  
  for (i in 1:num_round) {
    
    Iterator$reset()
    batch_loss = list()
    if (save.grad) {batch_grad = list()}
    batch_seq = 0
    t0 = Sys.time()
    
    while (Iterator$iter.next()) {
      
      my_values <- Iterator$value()
      mx.exec.update.arg.arrays(my_executor, arg.arrays = my_values, match.name = TRUE)
      mx.exec.forward(my_executor, is.train = TRUE)
      mx.exec.backward(my_executor)
      update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
      mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
      batch_loss[[length(batch_loss) + 1]] = as.array(my_executor$ref.outputs[[1]])
      if (save.grad) {
        grad_list = sapply(my_executor$ref.grad.arrays, function (x) {if (!is.null(x)) {mean(abs(as.array(x)))}})
        grad_list = unlist(grad_list[grepl('weight', names(grad_list), fixed = TRUE)])
        batch_grad[[length(batch_grad) + 1]] = grad_list
      }
      batch_seq = batch_seq + 1
      
    }
    
    message(paste0("epoch = ", i,
                   ": loss = ", formatC(mean(unlist(batch_loss)), format = "f", 4),
                   " (Speed: ", formatC(batch_seq * batch_size/as.numeric(Sys.time() - t0, units = 'secs'), format = "f", 2), " sample/secs)"))
    
    if (save.grad) {epoch_grad = rbind(epoch_grad, apply(abind(batch_grad, along = 2), 1, mean))}
    
  }
  
  if (save.grad) {
    
    epoch_grad[epoch_grad < 1e-8] = 1e-8
    
    COL = rainbow(ncol(epoch_grad))
    random_pos = 2^runif(ncol(epoch_grad), -0.5, 0.5)
    
    plot(epoch_grad[,1] * random_pos[1], type = 'l', col = COL[1],
         xlab = 'epoch', ylab = 'mean of abs(grad)', log = 'y',
         ylim = range(epoch_grad))
    
    for (i in 2:ncol(epoch_grad)) {lines(1:nrow(epoch_grad), epoch_grad[,i] * random_pos[i], col = COL[i])}
    
    legend('topright', paste0('layer', 1:ncol(epoch_grad), '_weight'), col = COL, lwd = 1)
    
  }
  
  #5. Get model
  my_model <- mxnet:::mx.model.extract.model(symbol = pred_symbol,
                                             train.execs = list(my_executor))
  
  return(my_model)
  
}