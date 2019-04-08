# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import os
import sys
import urllib.request

import numpy as np
import tensorflow as tf

class CIFAR10:
    H, W, C = 32, 32, 3
    LABELS = 10

    MEAN = [0.3640889, 0.3640889, 0.3640889]
    STD = [0.25968078, 0.25248665, 0.2553435]

    CUTOUT_PROB = 0.5
    CUTOUT_SIZE = 16

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/cifar10_competition.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, augment, sparse_labels=True, seed=42):
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])
            self._use_augmentation = augment

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

            self._data["images"] = CIFAR10.normalize(self._data["images"])
            if not sparse_labels:
                self._data["labels"][self._data["labels"] == 255] = 0
                self._data["labels"] = tf.keras.utils.to_categorical(self._data["labels"], num_classes=CIFAR10.LABELS)

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            while True:
                permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    permutation = permutation[batch_size:]

                    images = self._data["images"][batch_perm]
                    labels = self._data["labels"][batch_perm]

                    if self._use_augmentation:
                        images = np.array(list(map(CIFAR10.augment, images)))

                    yield (images, labels)


    def __init__(self, sparse_labels=True):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading CIFAR-10 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        cifar = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], cifar[key]) for key in cifar if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train", augment=dataset == "train", sparse_labels=sparse_labels))

    @staticmethod
    def augment(image):
        return CIFAR10.horizontal_flip(CIFAR10.cutout(CIFAR10.translate(image)))

    @staticmethod
    def cutout(image):
        if np.random.uniform() > CIFAR10.CUTOUT_PROB: return image
        half_size = CIFAR10.CUTOUT_SIZE // 2

        left = np.random.randint(-half_size, image.shape[0] - half_size)
        top = np.random.randint(-half_size, image.shape[1] - half_size)

        image[max(0, left):min(image.shape[0], left + CIFAR10.CUTOUT_SIZE),
        max(0, top):min(image.shape[1], top + CIFAR10.CUTOUT_SIZE), :] = 0
        return image

    @staticmethod
    def horizontal_flip(image):
        if np.random.uniform() > 0.5: return image
        return np.fliplr(image)

    @staticmethod
    def translate(image, amount=4):
        clamp = lambda n: max(0, min(n, CIFAR10.H))
        top, left = np.random.randint(-amount, amount), np.random.randint(-amount, amount)

        translated = np.zeros((CIFAR10.H, CIFAR10.W, CIFAR10.C))
        translated[clamp(-top):clamp(-top + CIFAR10.H), clamp(-left):clamp(-left + CIFAR10.W), :] = \
            image[clamp(top):clamp(top + CIFAR10.H), clamp(left):clamp(left + CIFAR10.W), :]

        return translated

    @staticmethod
    def normalize(image):
        return (image - CIFAR10.MEAN) / CIFAR10.STD