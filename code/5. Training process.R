
source('code/1. Iterator.R')
source('code/2. Optimizer.R')
source('code/3. Training function.R')
source('code/4. Model architecture.R')

Capsnet_model = my.model.FeedForward.create(Iterator = my_iter, ctx = mx.gpu(), save.grad = FALSE,
                                            loss_symbol = final_loss, pred_symbol = final_loss,
                                            Optimizer = my_optimizer, num_round = 30)


mx.model.save(Capsnet_model, 'CapsNet', 0)
