# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

from wide_res_net import WideResNet
from cifar10_augmented import CIFAR10

import numpy as np
import argparse


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_augmentations", default=1, type=int, help="Number of augmentations to average before the final prediction")
    parser.add_argument("--model_path", default='models/wideresnet_28-10_acc=0.9702', type=str, help="Path to weights of a model.")
    parser.add_argument("--output_path", default='dev_out.txt', type=str, help="Path to test predictions.")
    parser.add_argument("--depth", default=28, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=10, type=int, help="Widening factor over classical resnet.")
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    args = parser.parse_args()

    # Load data
    cifar = CIFAR10(sparse_labels=False)

    # Create the network
    network = WideResNet(args.depth, args.width_factor, args.weight_decay)
    network.load_weights(args.model_path)
    predicted_labels = np.argmax(network.predict_augmented(cifar.dev.data["images"], args.predict_augmentations), axis=1)

    print(np.mean(np.equal(np.argmax(cifar.dev.data["labels"], axis=1), np.array(predicted_labels))))

    with open(args.output_path, "w", encoding="utf-8") as out_file:
        print(*predicted_labels, file=out_file, sep='\n')
