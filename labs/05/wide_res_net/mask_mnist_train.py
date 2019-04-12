# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import tensorflow as tf

from u_wide_res_net import WideResNet
from mnist_augmented_masks import MNIST
from cyclic_learning_rate import CyclicLearningRate
from sgdr_learning_rate import SGDRLearningRate

import argparse
import os
import math

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=10, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=1, type=int, help="Widening factor over classical resnet.")
    # the weight decay is divided by two because: https://bbabenko.github.io/weight-decay/
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--lr_type", default='sgdr', type=str, help="Use cyclic learning rate or constant lr from the paper.")
    args = parser.parse_args()

    checkpoint_path = os.path.join("models", "wideresnet_{}-{}_{}_{}".format(args.depth, args.width_factor, "acc={val_class_output_accuracy:.4f}", "iou={val_mask_output_mean_iou:.4f}"))

    # Load data
    mnist = MNIST(sparse_labels=args.label_smoothing == 0)

    # Create the network and train
    network = WideResNet(args.depth, args.width_factor, args.weight_decay)

    train_step = math.ceil(mnist.train.size / args.batch_size)
    if args.lr_type == 'sgdr':
        learning_rate = SGDRLearningRate(learning_rate=0.125, t_0=40.0*train_step, m_mul=0.8)
    elif args.lr_type == 'cyclic':
        learning_rate = CyclicLearningRate(0.001, 0.2, step_size=20.0 * train_step)
    else:
        learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
            [train_step * 60.0, train_step * 120.0, train_step * 160.0],
            [0.1 * (0.2 ** i) for i in range(4)]
        )

    network.train(checkpoint_path, mnist, args.batch_size, args.epochs, args.label_smoothing, learning_rate)
