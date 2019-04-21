#!/usr/bin/env python3
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from caltech42_crossvalidation import Caltech42
import caltech42_augmentor as aug

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class SpickaNet(tf.keras.Sequential):
    def __init__(self):
        super(SpickaNet, self).__init__([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(Caltech42.LABELS, activation=None)
        ])


class Network:
    def __init__(self, args):
        
        self.models = []
        self.spickas = []

        for fold in range(args.folds):
            x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
            x = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
            
            spicka = SpickaNet()
            spicka.load_weights(args.model_path.format(fold))
            x = spicka(x)
        
            model = tf.keras.Model(inputs=inputs, outputs=x)
            self.models.append(model)
            self.spickas.append(spicka)        

    def test(self, caltech42, args):
        accuracy = 0.0
        for idx, dataset in enumerate(caltech42.folds):
            test_logs = self.models[idx].evaluate_generator(
                generator=dataset.dev.batches(args.batch_size, repeat=False),
                steps=dataset.dev.batched_size(args.batch_size),
                verbose=0
            )

            accuracy += test_logs[self.models[idx].metrics_names.index("accuracy")]

        print(accuracy / args.folds)

    def predict(self, caltech42, args):
        return self.model.predict_generator(
            generator=caltech42.train.data.batches(args.batch_size),
            steps=caltech42.train.batched_size(args.batch_size))


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model_path", default='models/acc-0.9640958189964295_fold-{}', type=str)
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--folds", default=10, type=int, help="Number of crossvalidation folds.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    caltech42 = Caltech42(aug.augment, aug.center_crop, sparse_labels=False, folds=args.folds)

    # Create the network and train
    network = Network(args)
    network.test(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
