#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

# Parse arguments
# TODO: Set reasonable defaults and possibly add more arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="tanh", type=str, help="Activation function.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--decay", default='exponential', type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=4, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

activation_dict = {"none": None, "relu": tf.nn.relu, "tanh": tf.nn.tanh, "sigmoid": tf.nn.sigmoid}


class LearningRateTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', profile_batch=2):
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch)

    def on_batch_end(self, batch, logs=None):
        logs.update({'learning rate': args.learning_rate if args.decay is None else float(model.optimizer.learning_rate(model.optimizer.iterations))})
        super().on_batch_end(batch, logs)


def learning_rate(decay):
    decay_steps = args.epochs * 100 // args.batch_size
    if decay is None:            return args.learning_rate
    elif decay == 'polynomial':  return tf.optimizers.schedules.PolynomialDecay(args.learning_rate, decay_steps, args.learning_rate_final)
    elif decay == 'exponential': return tf.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps, args.learning_rate_final / args.learning_rate)


def optimizer(type):
    if type == 'SGD':    return tf.optimizers.SGD(learning_rate=learning_rate(args.decay), momentum=0.0 if args.momentum is None else args.momentum)
    elif type == 'Adam': return tf.optimizers.Adam(learning_rate=learning_rate(args.decay))


# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load the data
observations, labels = [], []
with open("gym_cartpole-data.txt", "r") as data:
    for line in data:
        columns = line.rstrip("\n").split()
        observations.append([float(column) for column in columns[0:-1]])
        labels.append(int(columns[-1]))
observations, labels = np.array(observations), np.array(labels)

# TODO: Create the model in the `model` variable.
# However, beware that there is currently a bug in Keras which does
# not correctly serialize InputLayer. Instead of using an InputLayer,
# pass explicitly `input_shape` to the first real model layer.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(args.hidden_layer, activation=activation_dict[args.activation], input_shape=(4,)),
    #tf.keras.layers.Dense(args.hidden_layer, activation=activation_dict[args.activation]),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

model.compile(
    optimizer=optimizer(args.optimizer),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback = LearningRateTensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])

model.save("gym_cartpole_model.h5", include_optimizer=False)
