import math
import tensorflow as tf
import matplotlib.pyplot as plt


@tf.function
def constant_schedule_with_warmup(epoch, warmup_epochs=0, lr_start=1e-4, lr_max=1e-3):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between {lr_start} and {lr_max}.
    """

    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    else:
        lr = lr_max

    return lr

@tf.function
def linear_schedule_with_warmup(epoch, total_epochs, warmup_epochs=0, hold_max_epochs=0, 
                                    lr_start=1e-4, lr_max=1e-3, lr_min=None):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    
    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    elif epoch < warmup_epochs + hold_max_epochs:
        lr = lr_max
    else:
        lr = lr_max * ((total_epochs - epoch) / (total_epochs - warmup_epochs - hold_max_epochs))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

@tf.function
def cosine_schedule_with_warmup(epoch, total_epochs, warmup_epochs=0, hold_max_epochs=0, 
                                    lr_start=1e-4, lr_max=1e-3, lr_min=None, num_cycles=0.5):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    
    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    elif epoch < warmup_epochs + hold_max_epochs:
        lr = lr_max
    else:
        progress = (epoch - warmup_epochs - hold_max_epochs) / (total_epochs - warmup_epochs - hold_max_epochs)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(math.pi * num_cycles * 2.0 * progress)))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)
            
    return lr

@tf.function
def cosine_with_hard_restarts_schedule_with_warmup(epoch, total_epochs, warmup_epochs=0, lr_start=1e-4, lr_max=1e-3, lr_min=1e-4, num_cycles=1.):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """
    
    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(math.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

@tf.function
def exponential_schedule_with_warmup(epoch, warmup_epochs=0, hold_max_epochs=0, lr_start=1e-4, 
                                         lr_max=1e-3, lr_min=None, decay=0.9):
    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    elif epoch < warmup_epochs + hold_max_epochs:
        lr = lr_max
    else:
        lr = lr_max * (decay ** (epoch - warmup_epochs - hold_max_epochs))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)
            
    return lr

def step_schedule_with_warmup(epoch, step_size, warmup_epochs=0, hold_max_epochs=0, lr_start=1e-4, lr_max=1e-3, step_decay=.5):
    """ Create a schedule with a step decrease preceded by a warmup
        period during which the learning rate increases linearly between {lr_start} and {lr_max}.
    """

    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    elif epoch < warmup_epochs + hold_max_epochs:
        lr = lr_max
    else:
        lr = lr_max * step_decay**((epoch - warmup_epochs - hold_max_epochs)//step_size)
    return lr


# Usage example
### Cosine decay with warmup schedule

# n_epochs = 50
# rng = [i for i in range(n_epochs)]
# y = [cosine_schedule_with_warmup(tf.cast(x, tf.float32), total_epochs=n_epochs, warmup_epochs=3) for x in rng]
# fig, ax = plt.subplots(figsize=(20, 4))
# plt.plot(rng, y)
# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))