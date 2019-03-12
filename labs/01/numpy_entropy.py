#!/usr/bin/env python3
from collections import Counter

import numpy as np

if __name__ == "__main__":

    # Load data distribution, each data point on a line
    counter = Counter()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            counter[line] += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.
    data_dist = np.array([counter[k] for k in counter.keys()])
    data_dist = data_dist / sum(data_dist)

    # Load model distribution, each line `word \t probability`.
    model_dictionary = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n").split()
            # TODO: process the line, aggregating using Python data structures
            model_dictionary[line[0]] = float(line[1])

    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array([model_dictionary[k] if (k in model_dictionary) else 0.0 for k in counter.keys()])

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_dist*np.log(data_dist))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = -np.sum(data_dist*np.log(model_dist))
    print("{:.2f}".format(cross_entropy))

    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(kl_divergence))
