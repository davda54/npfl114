# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.layers import BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from cifar10_augmented import CIFAR10


class WideResNet(tf.keras.Model):
    def __init__(self, depth, width_factor, weight_decay):
        self.depth = depth
        self.width_factor = width_factor
        self.weight_decay = weight_decay

        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

    def train(self, checkpoint_path, data, batch_size, num_epochs, label_smoothing):
        train_step = data.train.size / batch_size
        learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
            [train_step * 60, train_step * 120, train_step * 160],
            [0.1 * (0.2 ** i) for i in range(4)]
        )
        self.compile(
            optimizer=tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=self._loss(label_smoothing),
            metrics=[self._metric(label_smoothing)],
        )
        self.fit_generator(
            generator=data.train.batches(batch_size),
            steps_per_epoch=math.ceil(data.train.size / batch_size),
            epochs=num_epochs,
            validation_data=data.dev.batches(batch_size),
            validation_steps=math.ceil(data.dev.size / batch_size),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,
                                                          save_weights_only=True)]
        )

    def predict_augmented(self, input, augmentation_loops):
        labels = []
        for image in input:
            ensamble = [CIFAR10.horizontal_flip(CIFAR10.translate(image, amount=4)) for _ in range(augmentation_loops)]
            predictions = tf.nn.softmax(self.predict(np.array(ensamble)))
            labels.append(np.sum(predictions, axis=0))

        return np.array(labels)

    def _build(self):
        filters = [16, 1*16 * self.width_factor, 2*16 * self.width_factor, 4*16 * self.width_factor]
        block_depth = (self.depth - 4) // (3 * 2)

        x = inputs = Input(shape=(CIFAR10.H, CIFAR10.W, CIFAR10.C), dtype=tf.float32)
        x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)

        x = self._block(x, stride=1, depth=block_depth, filters=filters[1])  # No pooling
        x = self._block(x, stride=2, depth=block_depth, filters=filters[2])  # Puling ve dvi, should be 16x16
        x = self._block(x, stride=2, depth=block_depth, filters=filters[3])  # Puling ve dvi, should be 8x8

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(CIFAR10.LABELS, activation=None)(x)

        return inputs, outputs

    def _block(self, x, stride, depth, filters):
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
        if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

    def _metric(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        return tf.keras.metrics.CategoricalAccuracy(name="accuracy")


if __name__ == "__main__":
    import argparse
    import os

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, type=bool, help="Train or just predict?")
    parser.add_argument("--predict_augmentations", default=1, type=int, help="Number of augmentations of test data before the final prediction")
    parser.add_argument("--model_path", default='wideresnet_models/tlustoprd_40-4_acc=0.9674', type=str, help="Path to weights of a model.")
    parser.add_argument("--ensamble_directory", default='wideresnet_models', type=str, help="Path to weights of a model.")
    parser.add_argument("--output_path", default='dev_out.txt', type=str, help="Path to test predictions.")
    parser.add_argument("--depth", default=40, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=4, type=int, help="Widening factor over classical resnet.")
    # the weight decay is divided by two because: https://bbabenko.github.io/weight-decay/
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Evaluation batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load data
    cifar = CIFAR10(sparse_labels=args.label_smoothing == 0)

    # Create the network
    network = WideResNet(args.depth, args.width_factor, args.weight_decay)

    if args.train:
        checkpoint_path = os.path.join("wideresnet_models", "tlustoprd_28-10_{}".format("acc={val_accuracy:.4f}"))
        network.train(checkpoint_path, cifar, args.batch_size, args.epochs, args.label_smoothing)

    network.load_weights(args.model_path)
    predicted_labels = network.predict_augmented(cifar.dev.data["images"], args.predict_augmentations)

    print(np.mean(np.equal(np.argmax(cifar.dev.data["labels"], axis=1), np.array(predicted_labels))))

    with open(args.output_path, "w", encoding="utf-8") as out_file:
        print(*predicted_labels, file=out_file, sep='\n')

