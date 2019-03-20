#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="elu", type=str, help="Activation function.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
parser.add_argument("--decay", default='exponential', type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Helper methods
activation_dict = {
    "none": None,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "elu": tf.nn.elu,
    "leaky_relu": tf.nn.leaky_relu
}

def learning_rate(decay, train_size, lr, lr_final):
    decay_steps = args.epochs * train_size // args.batch_size
    if decay is None:
        return lr
    elif decay == 'polynomial':
        return tf.optimizers.schedules.PolynomialDecay(lr, decay_steps, lr_final)
    elif decay == 'exponential':
        return tf.optimizers.schedules.ExponentialDecay(lr, decay_steps, lr_final / lr)

def sgd_optimizer(type, train_size, lr, lr_final):
    if type == 'SGD':
        return tf.optimizers.SGD(learning_rate=learning_rate(args.decay, train_size, lr, lr_final), momentum=0.0 if args.momentum is None else args.momentum)
    elif type == 'Adam':
        return tf.optimizers.Adam(learning_rate=learning_rate(args.decay, train_size, lr, lr_final))

def sparse_to_distribution(labels, label_smoothing):
    if label_smoothing == 0: return labels
    return np.eye(2)[labels]

def loss(label_smoothing):
    if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

def metric(label_smoothing):
    if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    return tf.keras.metrics.CategoricalAccuracy(name="accuracy")


# Black-box function to be optimized by Mr. Bayes
def model_accuracy(alphabet_size, dropout, embedding_size, layer_size, label_smoothing, learning_rate, learning_rate_final, window):

    # Scale the parameters correctly
    alphabet_size = int(alphabet_size + 0.5)
    embedding_size = min(alphabet_size, int(embedding_size + 0.5))
    layer_size = int(layer_size + 0.5)
    learning_rate = 10 ** learning_rate
    learning_rate_final = max(learning_rate, 10 ** learning_rate_final)
    window = int(window + 0.5)

    # Load data
    uppercase_data = UppercaseData(window, alphabet_size)

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(alphabet_size, embedding_size, input_length=2*window + 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(layer_size, activation=activation_dict[args.activation]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(uppercase_data.LABELS)
    ])

    model.compile(
        optimizer=sgd_optimizer(args.optimizer, uppercase_data.train.size, learning_rate, learning_rate_final),
        loss=loss(label_smoothing),
        metrics=[metric(label_smoothing)],
    )

    model.fit(
        uppercase_data.train.data["windows"],
        sparse_to_distribution(uppercase_data.train.data["labels"], label_smoothing),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0
    )

    test_logs = model.evaluate(
        uppercase_data.dev.data["windows"],
        sparse_to_distribution(uppercase_data.dev.data["labels"], label_smoothing),
        batch_size=args.batch_size
    )

    return test_logs[model.metrics_names.index("accuracy")]


pbounds = {
    'alphabet_size': (64.0, 128.0),
    'dropout': (0.3, 0.5),
    'embedding_size': (16.0, 64.0),
    'layer_size': (512.0, 1024.0),
    'label_smoothing': (0.0, 0.2),
    'learning_rate': (-4.0, -1.5),
    'learning_rate_final': (-5.0, -3),
    'window': (4.5, 8.0)
}

optimizer = BayesianOptimization(
    f=model_accuracy,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

logger = JSONLogger(path="./parameters_log.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)
