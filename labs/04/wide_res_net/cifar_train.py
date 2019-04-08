# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

from wide_res_net import WideResNet
from cifar10_augmented import CIFAR10

import argparse
import os


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=28, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=10, type=int, help="Widening factor over classical resnet.")
    # the weight decay is divided by two because: https://bbabenko.github.io/weight-decay/
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    args = parser.parse_args()

    checkpoint_path = os.path.join("models", "wideresnet_{}-{}_{}".format(args.depth, args.width_factor, "acc={val_accuracy:.4f}"))

    # Load data
    cifar = CIFAR10(sparse_labels=args.label_smoothing == 0)

    # Create the network and train
    network = WideResNet(args.depth, args.width_factor, args.weight_decay)
    network.train(checkpoint_path, cifar, args.batch_size, args.epochs, args.label_smoothing)
