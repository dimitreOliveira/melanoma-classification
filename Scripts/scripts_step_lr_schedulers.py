import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


@tf.function
def constant_schedule_with_warmup(step, warmup_steps=0, lr_start=1e-4, lr_max=1e-3):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between {lr_start} and {lr_max}.
    """

    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    else:
        lr = lr_max

    return lr

@tf.function
def linear_schedule_with_warmup(step, total_steps, warmup_steps=0, hold_max_steps=0, 
                                lr_start=1e-4, lr_max=1e-3, lr_min=None):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        lr = lr_max * ((total_steps - step) / (total_steps - warmup_steps - hold_max_steps))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

@tf.function
def cosine_schedule_with_warmup(step, total_steps, warmup_steps=0, hold_max_steps=0, 
                                lr_start=1e-4, lr_max=1e-3, lr_min=None, num_cycles=0.5):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / (warmup_steps * step + lr_start)
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        progress = (step - warmup_steps - hold_max_steps) / (total_steps - warmup_steps - hold_max_steps)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(np.pi * num_cycles * progress)))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)
            
    return lr

@tf.function
def cosine_with_hard_restarts_schedule_with_warmup(step, total_steps, warmup_steps=0, lr_start=1e-4, 
                                                   lr_max=1e-3, lr_min=1e-4, num_cycles=1.):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(np.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)

    return lr

@tf.function
def exponential_schedule_with_warmup(step, warmup_steps=0, hold_max_steps=0, lr_start=1e-4, 
                                     lr_max=1e-3, lr_min=None, decay=0.9):
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        lr = lr_max * (decay ** (step - warmup_steps - hold_max_steps))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)
            
    return lr

@tf.function
def one_cycle_schedule(step, total_steps, warmup_steps=None, hold_max_steps=0, lr_start=1e-4, lr_max=1e-3):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    if warmup_steps is None:
      warmup_steps = (total_steps - hold_max_steps) // 2
    
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        current_percentage = step / total_steps
        if current_percentage <= .9:
            lr = lr_max * ((total_steps - step) / (total_steps - warmup_steps - hold_max_steps))
        else:
            lr = lr_max * ((total_steps - step) / (total_steps - warmup_steps - hold_max_steps) * 0.8)

    return lr

@tf.function
def one_cycle_schedule_inv(step, total_steps, warmup_steps=None, hold_min_steps=0, m_max=.95, m_min=.85):
    """ Create a schedule with a momentum that increases linearly after
    linearly decreasing during a warmup period.
    """
    if warmup_steps is None:
        warmup_steps = (total_steps - hold_min_steps) // 2
    
    if step < warmup_steps:
        m = m_max + (step - 0) * ((m_min - m_max) / (warmup_steps - 0))
    elif step < warmup_steps + hold_min_steps:
        m = m_min
    else:
        current_percentage = step / total_steps
        if current_percentage <= .9:
            m = m_min + (step - warmup_steps - hold_min_steps) * ((m_max - m_min) / ((total_steps * .9) - warmup_steps - hold_min_steps))
        else:
            m = m_max

    return m

@tf.function
def step_schedule_with_warmup(step, step_size, warmup_steps=0, hold_max_steps=0, lr_start=1e-4, lr_max=1e-3, step_decay=.5):
    """ Create a schedule with a step decrease preceded by a warmup
        period during which the learning rate increases linearly between {lr_start} and {lr_max}.
    """

    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    elif step < warmup_steps + hold_max_steps:
        lr = lr_max
    else:
        lr = lr_max * step_decay**((step - warmup_steps - hold_max_steps)//step_size)
    return lr


#Usage example
### Exponential decay with warmup schedule

# n_steps = 500
# rng = [i for i in range(n_epochs)]

# y = [exponential_schedule_with_warmup(tf.cast(x, tf.float32), lr_min=1e-4, decay=0.995) for x in rng]
# fig, ax = plt.subplots(figsize=(20, 4))
# plt.plot(rng, y)
# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


class LRFinder(Callback):
    ''' 
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
        source: https://gist.github.com/jeremyjordan/ac0229abd4b2b7000aca1643e88e0f02
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()