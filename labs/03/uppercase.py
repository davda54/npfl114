#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation function.")
parser.add_argument("--alphabet_size", default=80, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
parser.add_argument("--decay", default='exponential', type=str, help="Learning decay rate type")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout regularization.")
parser.add_argument("--embedding_size", default=64, type=float, help="Dimension of the embedding layer.")
parser.add_argument("--embedding_dropout", default=0.0, type=float, help="Embedding dropout regularization.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="512", type=str, help="Hidden layer configuration.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=6, type=int, help="Window size to use.")

args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

def learning_rate(decay):
    decay_steps = args.epochs * uppercase_data.train.size // args.batch_size
    if decay is None: return args.learning_rate
    elif decay == 'polynomial': return tf.optimizers.schedules.PolynomialDecay(args.learning_rate, decay_steps, args.learning_rate_final)
    elif decay == 'exponential': return tf.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps, args.learning_rate_final / args.learning_rate)

def optimizer(type):
    if type == 'SGD':return tf.optimizers.SGD(learning_rate=learning_rate(args.decay), momentum=0.0 if args.momentum is None else args.momentum)
    elif type == 'Adam': return tf.optimizers.Adam(learning_rate=learning_rate(args.decay))

def sparse_to_distribution(labels):
    return np.eye(uppercase_data.LABELS)[labels]

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(args.alphabet_size, args.embedding_size, input_length=2*args.window + 1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(args.embedding_dropout))
for hidden_layer in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden_layer, activation=args.activation))
    model.add(tf.keras.layers.Dropout(args.dropout))
model.add(tf.keras.layers.Dense(uppercase_data.LABELS))

model.compile(
    optimizer=optimizer(args.optimizer),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy") if args.label_smoothing > 0 else tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    uppercase_data.train.data["windows"], sparse_to_distribution(uppercase_data.train.data["labels"]) if args.label_smoothing > 0 else uppercase_data.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(uppercase_data.dev.data["windows"], sparse_to_distribution(uppercase_data.dev.data["labels"]) if args.label_smoothing > 0 else uppercase_data.dev.data["labels"]),
    callbacks=[tb_callback],
)


with open("uppercase_test.txt", "w") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    pass
