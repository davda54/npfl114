# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

from wide_res_net import WideResNet
from cifar10_augmented import CIFAR10

import numpy as np
import argparse
import os


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_augmentations", default=64, type=int, help="Number of augmentations to average before the final prediction")
    parser.add_argument("--ensamble_directory", default='models', type=str, help="Path to the directory with weights of ensamble models.")
    parser.add_argument("--output_path", default='test_out.txt', type=str, help="Path to test predictions.")
    parser.add_argument("--depth", default=28, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=10, type=int, help="Widening factor over classical resnet.")
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    args = parser.parse_args()

    # Load data
    cifar = CIFAR10(sparse_labels=False)

    # Create the network
    network = WideResNet(args.depth, args.width_factor, args.weight_decay)

    ensamble_predictions = np.zeros((cifar.test.size, CIFAR10.LABELS))
    # Process each model, note that the weights are saved in two files and one weight file differs just by the suffix after '.'
    for weights in {s[:len("wideresnet_{}-{}_acc=X.XXXX".format(args.depth, args.width_factor))] for s in os.listdir(args.ensamble_directory)}:
        if weights[0] == '.': continue
            
        network.load_weights(os.path.join(args.ensamble_directory, weights))
        ensamble_predictions += network.predict_augmented(cifar.test.data["images"], args.predict_augmentations)

    predicted_labels = np.argmax(ensamble_predictions, axis=1)

    with open(args.output_path, "w", encoding="utf-8") as out_file:
        print(*predicted_labels, file=out_file, sep='\n')
