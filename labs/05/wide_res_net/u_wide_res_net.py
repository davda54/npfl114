# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Pro segmentaci používáme U-Net architekturu, ve které je jako základ použit Wide-Res-Net.
# U klasifikace se nakonec ukázalo vhodnější použít samostatnou WRN síť na vstupy zamaskované pomocí segmentační sítě
# Regularizujeme augmentací vstupu (horizontální zrdcadlení a posunutí), label smoothingu, l2 a cutoutu
# Výsledek je ensamble zhruba deseti nejlepších checkpointů

import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.layers import Layer, BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input, add, UpSampling2D, Conv2DTranspose
from tensorflow.keras.regularizers import l2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from mnist_augmented_masks import MNIST
from iou_metric import IoUMetric

class UWideResNet(tf.keras.Model):
    def __init__(self, depth, width_factor, weight_decay):
        self.depth = depth
        self.width_factor = width_factor
        self.weight_decay = weight_decay

        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

    def train(self, checkpoint_path, data, batch_size, num_epochs, label_smoothing, learning_rate):
        self.compile(
            optimizer=tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=self._loss(label_smoothing),
            metrics=self._metric(label_smoothing),
        )
        self.fit_generator(
            generator=data.train.batches(batch_size),
            steps_per_epoch=math.ceil(data.train.size / batch_size),
            epochs=num_epochs,
            validation_data=data.dev.batches(batch_size),
            validation_steps=math.ceil(data.dev.size / batch_size),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_class_output_accuracy', save_best_only=False,
                                                          save_weights_only=True)],
            verbose=2
        )

    def _build(self):
        filters = [16, 1 * 16 * self.width_factor, 2 * 16 * self.width_factor, 4 * 16 * self.width_factor]
        down_block_depth = (self.depth - 4) // (3 * 2)
        up_block_depth = down_block_depth

        x = inputs = Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)

        x_1 = self._down_block(x, stride=1, depth=down_block_depth, filters=filters[1])
        x_2 = self._down_block(x_1, stride=2, depth=down_block_depth, filters=filters[2])
        x_3 = self._down_block(x_2, stride=2, depth=down_block_depth, filters=filters[3])

        mask_x_1 = self._up_block(x_3, x_2, depth=up_block_depth, filters=filters[2])
        mask_x_2 = self._up_block(mask_x_1, x_1, depth=up_block_depth, filters=filters[1])

        class_x = BatchNormalization()(x_3)
        class_x = ReLU()(class_x)
        class_x = GlobalAveragePooling2D()(class_x)
        class_prediction = Dense(MNIST.LABELS, activation=None, name='class_output')(class_x)

        mask_prediction = BatchNormalization()(mask_x_2)
        mask_prediction = ReLU()(mask_prediction)
        mask_prediction = Conv2D(filters=1, kernel_size=(1,1), activation=tf.nn.sigmoid,
                                 kernel_initializer='he_normal', name='mask_output')(mask_prediction)

        return inputs, [class_prediction, mask_prediction]

    def _up_block(self, x, skip, depth, filters):
        x = self._upsample_layer(x, skip, filters, stride=2)
        for _ in range(depth - 1):
            x = self._basic_layer(x, filters)
        return x

    def _down_block(self, x, stride, depth, filters):
        x = self._downsample_layer(x, filters, stride)
        for _ in range(depth - 1):
            x = self._basic_layer(x, filters)
        return x

    def _basic_layer(self, x, filters):
        block = tf.keras.Sequential([
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
        ])
        return block(x) + x

    def _upsample_layer(self, x, skip, filters, stride):
        x = BatchNormalization()(x)
        x = ReLU()(x)

        block = tf.keras.Sequential([
            Conv2DTranspose(filters, kernel_size=2, strides=stride, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
        ])
        upsample = tf.keras.Sequential([
            UpSampling2D((2, 2)),
            Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')
        ])

        return block(x) + upsample(x) + skip

    def _downsample_layer(self, x, filters, stride):
        x = BatchNormalization()(x)
        x = ReLU()(x)

        block = tf.keras.Sequential([
            Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
        ])
        downsample = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')

        return block(x) + downsample(x)

    def _loss(self, label_smoothing):
        if label_smoothing == 0: class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else: class_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
        mask_loss = tf.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        return [class_loss, mask_loss]

    def _metric(self, label_smoothing):
        if label_smoothing == 0: class_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        else: class_accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        mask_accuracy = IoUMetric(num_classes=2)
        return [[class_accuracy], [mask_accuracy]]