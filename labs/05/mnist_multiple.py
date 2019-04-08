#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.
        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation (the goal is to predict if the first digit is larger than the second)
        #
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.

        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=[tf.losses.SparseCategoricalCrossentropy(from_logits=True), tf.losses.SparseCategoricalCrossentropy(from_logits=True), tf.losses.BinaryCrossentropy()],
            metrics=[['accuracy'], ['accuracy'], ['accuracy']]
        )

    def _build(self):
        conv_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
        conv_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
        flatten = tf.keras.layers.Flatten()
        dense = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)
        classification_out = tf.keras.layers.Dense(units=10, activation=None)
        comparison_dense = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)
        comparision_out = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

        input_1 = tf.keras.Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        input_2 = tf.keras.Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)

        hidden_1 = dense(flatten(conv_2(conv_1(input_1))))
        hidden_2 = dense(flatten(conv_2(conv_1(input_2))))

        output_1 = classification_out(hidden_1)
        output_2 = classification_out(hidden_2)
        output_3 = comparision_out(comparison_dense(tf.concat([hidden_1, hidden_2], axis=1)))

        return [input_1, input_2], [output_1, output_2, output_3]

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                # TODO: yield the suitable modified inputs and targets using batches[0:2]
                model_inputs = [batches[0]["images"], batches[1]["images"]]
                model_targets = [batches[0]["labels"], batches[1]["labels"], batches[0]["labels"] > batches[1]["labels"]]
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for inputs, targets in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.train_on_batch(x=inputs, y=targets)

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        direct_accuracy = 0
        indirect_accuracy = 0

        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            prediction = self.predict(inputs)
            prediction = [np.argmax(prediction[0], axis=1), np.argmax(prediction[1], axis=1), prediction[2][:,0] > 0.5]
            direct_accuracy += np.sum(targets[2] == prediction[2])
            indirect_accuracy += np.sum(targets[2] == (prediction[0] > prediction[1]))

        return direct_accuracy / (dataset.size // 2), indirect_accuracy / (dataset.size // 2)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
