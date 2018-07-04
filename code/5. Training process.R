
source('code/1. Iterator.R')
source('code/2. Optimizer.R')
source('code/3. Training function.R')
source('code/4. Model architecture.R')

Capsnet_model = my.model.FeedForward.create(Iterator = my_iter, ctx = mx.gpu(), save.grad = FALSE,
                                            loss_symbol = m_logloss, pred_symbol = pred_out,
                                            Optimizer = my_optimizer, num_round = 20)


mx.model.save(Capsnet_model, 'CapsNet', 0)
