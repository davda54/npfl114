# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Pro segmentaci používáme U-Net architekturu, ve které je jako základ použit Wide-Res-Net.
# U klasifikace se nakonec ukázalo vhodnější použít samostatnou WRN síť na vstupy zamaskované pomocí segmentační sítě
# Regularizujeme augmentací vstupu (horizontální zrdcadlení a posunutí), label smoothingu, l2 a cutoutu
# Výsledek je ensamble zhruba deseti nejlepších checkpointů

from u_wide_res_net import UWideResNet
from wide_res_net import WideResNet
from mnist_augmented_masks import MNIST

import tensorflow as tf
import numpy as np
import argparse
import os


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensamble_directory", default='ensamble_models', type=str, help="Path to the directory with weights of ensamble models.")
    parser.add_argument("--output_path", default='test_out.txt', type=str, help="Path to test predictions.")
    args = parser.parse_args()

    # Load data
    mnist = MNIST(sparse_labels=False)

    ensamble_classifications = np.zeros((mnist.test.size, MNIST.LABELS))
    ensamble_masks = np.zeros((mnist.test.size, MNIST.H, MNIST.W, MNIST.C))

    # Process each UNET model to predict the masks, note that the weights are saved in two files and one weight file differs just by the suffix after '.'
    counter = 0
    for weights in {s[:len("wideresnet_XX-X_acc=X.XXXX_iou=X.XXXX")] for s in os.listdir(args.ensamble_directory)}:
        if weights[0] == '.': continue
        if 'iou=' not in weights: continue
        if float(weights.replace('_', '=').split('=')[5]) < 0.9974: continue
        
        depth, width_factor = [int(x) for x in weights.split('_')[1].split('-')]
        network = UWideResNet(depth, width_factor, 0.0005/2)
        network.load_weights(os.path.join(args.ensamble_directory, weights))

        _, masks = network.predict(mnist.test.data["images"])
        ensamble_masks += masks

        print("{}".format(counter))
        counter += 1

    ensamble_masks = np.round(ensamble_masks / counter)    
    masked_images = mnist.test.data["images"] * ensamble_masks
    
    print("DONE\n")
    for weights in {s[:len("wideresnet_XX-XX_acc=X.XXXX")] for s in os.listdir(args.ensamble_directory)}:
        if weights[0] == '.': continue
        if 'iou=' in weights: continue
        if float(weights.replace('_', '=').split('=')[3]) < 0.9570: continue
        
        depth, width_factor = [int(x) for x in weights.split('_')[1].split('-')]
        network = WideResNet(depth, width_factor, 0.0005/2)
        network.load_weights(os.path.join(args.ensamble_directory, weights))

        classifications = network.predict(masked_images)
        ensamble_classifications += classifications

        print("{}".format(counter))
        counter += 1
        
    ensamble_classifications = np.argmax(ensamble_classifications, axis=1)
    
    with open(args.output_path, "w", encoding="utf-8") as out_file:
        for label, mask in zip(ensamble_classifications, ensamble_masks):
            print(label, *mask.astype(np.uint8).flatten(), file=out_file)
