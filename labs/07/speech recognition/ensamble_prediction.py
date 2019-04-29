#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

from timit_mfcc import TimitMFCC


# Convert given sparse tensor to a (dense_output, sequence_lengths).
def _to_dense(tensor):
    tensor = tf.sparse.to_dense(tensor, default_value=-1)
    tensor_lens = tf.reduce_sum(tf.cast(tf.not_equal(tensor, -1), tf.int32), axis=1)
    return tensor, tensor_lens

# Perform CTC predictions given logits and their lengths.
def _ctc_predict(logits, logits_len):
    (predictions,), _ = tf.nn.ctc_beam_search_decoder(logits, logits_len, beam_width=256)
    return tf.cast(predictions, tf.int32)

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(e_x.sum(axis=-1), -1)


dataset = "test"
dir = "ensamble"
timit = TimitMFCC()


models = np.array([
    np.load(f"{dir}/BiGRU_4-256_acc-0.2787.{dataset}.npy"),  # 27.96 really
    np.load(f"{dir}/BiGRU_4-256_acc-0.2794.{dataset}.npy"),  # 28.05 really
    np.load(f"{dir}/BiGRU_4-256_acc-0.2802.{dataset}.npy"), # 27.97 really
    np.load(f"{dir}/BiGRU_4-256_acc-0.2807.{dataset}.npy"), # 28.07 really
    #np.load(f"{dir}/BiGRU_4-256_acc-0.2812.{dataset}.npy"),  # 28.10 really
    #np.load(f"{dir}/BiGRU_4-256_acc-0.2843.{dataset}.npy"), # 28.41 really
    #np.load(f"{dir}/BiGRU_4-256_acc-0.2846.{dataset}.npy"), # 28.42 really
    #np.load(f"{dir}/3-128-0.2783.h5_logits_{dataset}.npy") # 28.73 really
])

models = _softmax(models) # logits to probability
models = np.mean(models, axis=0) # ensamble
models = np.log(models) # inverse of softmax to go back to "logits"
models = np.transpose(models.astype(np.float32), [1, 0, 2]) # right format

print("ensambled")

lengths = np.array([sentence.shape[0] for sentence in getattr(timit, dataset).data["mfcc"]])
predictions = _ctc_predict(models, lengths)

sentences = []
for prediction, prediction_len in zip(*_to_dense(predictions)):
    sentences.append(prediction[:prediction_len])

out_path = f"speech_recognition_{dataset}.txt"
with open(out_path, "w", encoding="utf-8") as out_file:
    for sentence in sentences:
        print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)


# import os
# os.system("python speech_recognition_eval.py")