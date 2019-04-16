# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Pro segmentaci používáme U-Net architekturu, ve které je jako základ použit Wide-Res-Net.
# U klasifikace se nakonec ukázalo vhodnější použít samostatnou WRN síť na vstupy zamaskované pomocí segmentační sítě
# Regularizujeme augmentací vstupu (horizontální zrdcadlení a posunutí), label smoothingu, l2 a cutoutu
# Výsledek je ensamble zhruba deseti nejlepších checkpointů

import tensorflow as tf

class IoUMetric(tf.metrics.MeanIoU):

    def __init__(self, num_classes, name=None, dtype=None):
        super(IoUMetric, self).__init__(num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(IoUMetric, self).update_state(tf.math.round(y_true), tf.math.round(y_pred), sample_weight)