#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
from tqdm import tqdm

from caltech42_crossvalidation import Caltech42
from caltech42 import center_crop

# see https://arxiv.org/pdf/1412.6572.pdf for more on adversarial training
# see https://arxiv.org/pdf/1710.09412.pdf for more on mixup

# The neural network model
class Network:
    def __init__(self, args):
        x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
        x = bottlenecks = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
        
        dropout = tf.keras.layers.Dropout(0.42)
        dense = tf.keras.layers.Dense(Caltech42.LABELS, activation="softmax")
        
        x = dropout(x)
        x = probas = dense(x)
        
        self.bottlenecks = tf.keras.Model(inputs=inputs, outputs=bottlenecks)
        self.model = tf.keras.Model(inputs=inputs, outputs=probas)
        
        x = inputs = tf.keras.Input(shape=(1280,), dtype=tf.float32)
        x = dropout(x)
        x = dense(x)
        self.classifier = tf.keras.Model(inputs=inputs, outputs=x)
        
    @tf.function
    def adversarial_train_on_batch(self, batch_X, batch_y, alpha, epsilon, space):
        if space == "latent":
            batch_X = self.bottlenecks(batch_X, training=False)
            model = self.classifier
        else:
            model = self.model
        
        with tf.GradientTape() as tape:
            tape.watch(batch_X)
            probas = model(batch_X, training=True)
            loss = self.loss(batch_y, probas)
        input_gradients, batch_gradients = tape.gradient(loss, (batch_X, model.trainable_variables))
        
        self.train_acc(batch_y, probas)
        
        # apply original batch gradients here - Nesterov-ish way
        self.optimizer.apply_gradients(zip(batch_gradients, self.model.trainable_variables))
        
        adversarial_X = batch_X + epsilon * tf.sign(input_gradients)
        with tf.GradientTape() as tape:
            probas = model(adversarial_X, training=True)
            loss = self.loss(batch_y, probas)
        adversarial_gradients = tape.gradient(loss, model.trainable_variables)

        self.adversarial_acc(batch_y, probas)
        
        self.optimizer.apply_gradients(zip(adversarial_gradients, model.trainable_variables))
        
    @tf.function
    def mixup_train_on_batch(self, batch_X1, batch_y1, batch_X2, batch_y2, space):
        # weights = tf.random.uniform((batch_X1.shape[0],))
        weights = tf.abs(tf.random.truncated_normal((batch_X1.shape[0],), stddev=0.15))
        if space == "latent":
            batch_X1 = self.bottlenecks(batch_X1, training=False)
            batch_X2 = self.bottlenecks(batch_X2, training=False)
            model = self.classifier
            weights_y, weights_X = weights[:, None]
        else:
            model = self.model
            weights_X = weights[:, None, None, None]
            weights_y = weights[:, None]
        
        
        batch_X = weights_X * batch_X1 + (1 - weights_X) * batch_X2
        batch_y = weights_y * batch_y1 + (1 - weights_y) * batch_y2
        
        with tf.GradientTape() as tape:
            probas = model(batch_X, training=True)
            loss = self.loss(batch_y, probas)
        gradients = tape.gradient(loss, model.trainable_variables)
        
        self.train_acc(batch_y, probas)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    @tf.function
    def train_on_batch(self, batch_X, batch_y):
        with tf.GradientTape() as tape:
            probas = self.model(batch_X, training=True)
            loss = self.loss(batch_y, probas)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        self.train_acc(batch_y, probas)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
    def train(self, caltech42, args):
        self.optimizer = tf.optimizers.Adam()
        if args.mode == "mixup":
            self.loss = tf.losses.CategoricalCrossentropy()
            self.train_acc = tf.metrics.CategoricalAccuracy()
            self.val_acc = tf.metrics.CategoricalAccuracy()
        else:  # args.mode in ("none", "adversarial")
            self.loss = tf.losses.SparseCategoricalCrossentropy()
            self.train_acc = tf.metrics.SparseCategoricalAccuracy()
            self.val_acc = tf.metrics.SparseCategoricalAccuracy()
            if args.mode == "adversarial":
                self.adversarial_acc = tf.metrics.SparseCategoricalAccuracy()
        
        best_acc = 0
        for epoch_i in range(args.epochs):
            self.train_acc.reset_states()
            
            if args.mode == "adversarial":
                self.adversarial_acc.reset_states()
            
            # train
            if args.mode == "mixup":
                batches_1 = caltech42.train.batches(args.batch_size)
                batches_2 = caltech42.train.batches(args.batch_size)
                for (X1, y1), (X2, y2) in tqdm(
                    zip(batches_1, batches_2), leave=False,
                    total=caltech42.train.batched_size(args.batch_size)
                ):
                    self.mixup_train_on_batch(X1, y1, X2, y2, args.space)
            else:
                for (X, y) in tqdm(caltech42.train.batches(args.batch_size),
                                   leave=False, total=caltech42.train.batched_size(args.batch_size)):
                    if args.mode == "none":
                        self.train_on_batch(X, y)
                    else:  # args.mode == "adversarial"
                        self.adversarial_train_on_batch(
                            X, y, args.adversarial_alpha, args.adversarial_epsilon, args.space)
                
            # validate
            val_acc = self.evaluate(caltech42.dev, args)
            
            print('{:02d} - train {:.4f} - adv {:.4f} - val {:.4f}'.format(
                epoch_i, self.train_acc.result(),
                self.adversarial_acc.result() if (args.mode == "adversarial") else float("nan"),
                val_acc))
            
            if val_acc >= best_acc:
                best_acc = val_acc
                path = args.checkpoint_path.format(val_acc=best_acc)
                print("Saving {}".format(path))
                self.classifier.save_weights(path)
        
        return best_acc
            
    def evaluate(self, dataset, args):
        self.val_acc.reset_states()
        for (X, y) in dataset.batches(args.batch_size):
            probas = self.model(X, training=False)
            self.val_acc(y, probas)
        return self.val_acc.result()
    
    def predict(self, caltech42, args):
        return self.model.predict_generator(
            generator=caltech42.train.data.batches(args.batch_size),
            steps=caltech42.train.batched_size(args.batch_size))
    

def augment(image):
    # random crop
    y_margin, x_margin = np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE
    t, l = np.random.randint(y_margin + 1), np.random.randint(x_margin + 1)
    
    image = image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE)]
    
    # random flip
    if np.random.random() > 0.5:
        image = image[:, ::-1, :]
        
    return image
    

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer to use.")
    parser.add_argument("--mode", default="none", choices=("none", "adversarial", "mixup"),
                        help="Regularization technique to be used.")
    parser.add_argument("--adversarial_alpha", default=0.5, type=float, help="Adversarial loss intensity")
    parser.add_argument("--adversarial_epsilon", default=0.1, type=float,
                        help="Adversarial example intensity.")
    parser.add_argument("--space", default="input", choices=("input", "latent"),
                        help="Space of adversarial perturbation or feature mixup.")
    parser.add_argument("--k_folds", default=10, type=int, help="Cross-validation schema.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.gpu.set_per_process_memory_growth(True)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    checkpoint_dir = os.path.join(args.logdir, "weights")
    os.makedirs(checkpoint_dir)

    # Load data
    caltech42 = Caltech42(augment, center_crop, args.k_folds, sparse_labels=(args.mode != "mixup"), preserve_dev=True)

    # Create the network and train
#     best_fold_acc = 0
#     best_checkpoint = None
#     for fold_i, fold in enumerate(caltech42.folds):
#         args.checkpoint_path = os.path.join(checkpoint_dir, "_fold_{}_{{val_acc:.4f}}".format(fold_i))
#         network = Network(args)
#         fold_acc = network.train(fold, args)
#         print("Fold {} - acc {:.4f}".format(fold_i, fold_acc))
#         if fold_acc > best_fold_acc:
#             best_fold_acc = fold_acc
#             best_checkpoint = args.checkpoint_path.format(val_acc=best_fold_acc)
    
#     network.model.load_weights(best_checkpoint)
#     val_acc = network.evaluate(caltech42.dev, args)
#     network.model.save_weights(os.path.join(checkpoint_dir, "{:.4f}".format(val_acc)))

    best_checkpoints = []
    for fold_i, fold in enumerate(caltech42.folds):
        args.checkpoint_path = os.path.join(checkpoint_dir, "fold_{}_{{val_acc:.4f}}".format(fold_i))
        network = Network(args)
        fold_acc = network.train(fold, args)
        print("Fold {} - acc {:.4f}".format(fold_i, fold_acc))
        best_checkpoints.append(args.checkpoint_path.format(val_acc=fold_acc))
    
    x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    x = bottlenecks = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
    
    clfs = []
    for checkpoint_i, best_checkpoint in enumerate(best_checkpoints):
        dense = tf.keras.layers.Dense(Caltech42.LABELS, activation="softmax")
        
        clf_input = tf.keras.Input(shape=(1280,), dtype=tf.float32)
        clf_output = dense(clf_input)
        model = tf.keras.Model(inputs=clf_input, outputs=clf_output)
        model.load_weights(best_checkpoint)
        
        clfs.append(dense(x))
    
    outputs = tf.keras.layers.Average()(clfs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if args.mode == "mixup":
        model.compile("adam", tf.losses.CategoricalCrossentropy(), metrics=['acc'])
    else:  # args.mode in ("none", "adversarial")
        model.compile("adam", tf.losses.SparseCategoricalCrossentropy(), metrics=['acc'])

    _, val_acc = model.evaluate_generator(generator=caltech42.dev.batches(args.batch_size),
                                          steps=caltech42.dev.batched_size(args.batch_size), verbose=1)
    model.save_weights(os.path.join(checkpoint_dir, "{:.4f}".format(val_acc)))
        
