import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# based on https://github.com/mhmoodlan/cyclic-learning-rate/blob/master/clr.py,
# official source codes of tensorflow to rewrite it to TF2.0
# and cosine decay in https://github.com/tensorflow/tensorflow/blob/6fcbdc777fa5c8bef281d5f4e074397e44d4b09a/tensorflow/contrib/training/python/training/sgdr_learning_rate_decay.py

class CyclicLearningRate(LearningRateSchedule):

    def __init__(self, learning_rate=0.01, max_lr=0.1, step_size=20., gamma=0.99994, mode='triangular', name=None):
        super(CyclicLearningRate, self).__init__()

        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        self.name = name

    def __call__(self, step):
        with ops.name_scope(self.name, "CyclicLearningRate", [self.learning_rate, step]) as name:
            learning_rate = ops.convert_to_tensor(self.learning_rate, name="learning_rate")
            dtype = learning_rate.dtype
            step = math_ops.cast(step, dtype)
            step_size = math_ops.cast(self.step_size, dtype)
            max_lr = math_ops.cast(self.max_lr, dtype)

            # computing: cycle = floor( 1 + step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(step, double_step)
            cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

            # computing: x = abs( step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))

            # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)

            if self.mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(cycle - 1, tf.int32)), tf.float32))
            if self.mode == 'exp_range':
                gamma = math_ops.cast(self.gamma, dtype)
                clr = math_ops.multiply(math_ops.pow(gamma, step), clr)
            #if self.mode == 'cosine':

            return math_ops.add(clr, learning_rate, name=name)

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "max_lr": self.max_lr,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "mode": self.mode,
            "name": self.name
        }
