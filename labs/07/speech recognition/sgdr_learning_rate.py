#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import math

class SGDRLearningRate:

    def __init__(self, learning_rate, t_0, mul=1.0):
        self.learning_rate = learning_rate
        self.t_0 = t_0
        self.mul = mul

    def __call__(self, epoch):
        return self.value(epoch)        

    def value(self, epoch):
        x = epoch / self.t_0
        i_restart = int(x)
        x = x - i_restart
        base = self.learning_rate * (self.mul ** i_restart)

        return 0.5 * base * (math.cos(math.pi * x) + 1)