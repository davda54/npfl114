#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData
from uppercase_data_diakritika import UppercaseDataDiakritika


parser = argparse.ArgumentParser()
parser.add_argument("--models_path", default="./ensamble_models", type=str)
parser.add_argument("--output", default="", type=str)
args = parser.parse_args()

original_data = UppercaseData(9, 128)
p = np.zeros((original_data.dev.size, 2))

for filename in os.listdir(args.models_path):
    parameters = dict([pair.split('=') for pair in filename.split(',')])

    alphabet_size = int(parameters['a_s'])
    window_size = int(parameters['w'])
    if "diac" in parameters: uppercase_data = UppercaseDataDiakritika(window_size, alphabet_size)
    else: uppercase_data = UppercaseData(window_size, alphabet_size)

    model = tf.keras.models.load_model(os.path.join(args.models_path, filename), compile=False)
    print("Loaded: " + filename)
    model_p = model.predict(uppercase_data.dev.data["windows"])

    for i, c in enumerate(original_data.dev.text):
        if c.lower() not in uppercase_data.dev.alphabet: continue
        p[i] += model_p[i]


with open("dev_out.txt", "w", encoding="utf-8") as file:
    for i, c in enumerate(original_data.dev.text):
        c = c.lower()
        file.write(c.upper() if p[i, 0] < p[i, 1] and c in original_data.dev.alphabet else c)



