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

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--mode", default="none", choices=("none", "adversarial", "mixup"),
                        help="Regularization technique to be used.")
    parser.add_argument("--k_folds", default=10, type=int, help="Cross-validation schema.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()
    
    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.gpu.set_per_process_memory_growth(True)
    
    caltech42 = Caltech42(center_crop, center_crop, 10, sparse_labels=True, preserve_dev=True)
    
    if args.mode == "adversarial":
        best_checkpoints = [
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_0_0.9747",
#             "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_1_0.9596",
#             "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_2_0.9590",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_3_0.9691",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_4_0.9749",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_5_0.9752",
#             "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_6_0.9388",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_7_0.9697",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_202202-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=adversarial,o=adam,s=input,t=5/weights/fold_8_0.9837",
        ]
    elif args.mode == "mixup":
        best_checkpoints = [
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_0_0.9545",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_1_0.9646",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_2_0.9641",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_3_0.9691",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_4_0.9698",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_5_0.9653",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_6_0.9541",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_7_0.9747",
            "logs/caltech42_uber_benchmark_train.py-2019-04-22_214527-aa=0.5,ae=0.1,bs=32,e=32,kf=10,m=mixup,o=adam,s=input,t=5/weights/fold_8_0.9783"
        ]
    
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
    model.compile("adam", tf.losses.SparseCategoricalCrossentropy(), metrics=['acc'])

    _, val_acc = model.evaluate_generator(generator=caltech42.dev.batches(args.batch_size),
                                          steps=caltech42.dev.batched_size(args.batch_size), verbose=1)
    
    dev_predictions = model.predict_generator(generator=caltech42.dev.batches(args.batch_size),
                                              steps=caltech42.dev.batched_size(args.batch_size), verbose=1)
    test_predictions = model.predict_generator(generator=caltech42.test.batches(args.batch_size),
                                               steps=caltech42.test.batched_size(args.batch_size), verbose=1)
    
    np.save("{}_dev.npy".format(args.mode), dev_predictions)
    np.save("{}_test.npy".format(args.mode), test_predictions)