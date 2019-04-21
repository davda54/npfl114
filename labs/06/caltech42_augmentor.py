import random
import numpy as np
import tensorflow as tf
import scipy

from skimage.io import imread
from skimage.color import gray2rgb

from caltech42 import Caltech42

def augment(image):
    return horizontal_flip(random_crop(image))

def rotate(image):
    k = random.randint(0, 4)
    if k % 4 != 0: return tf.image.rot90(image, k)
    return image

def horizontal_flip(image):
    return tf.image.random_flip_left_right(image)

def random_crop(image):
    return tf.image.random_crop(image, [Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C])
#     t, l = np.round(np.random.uniform([0, 0], np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE)).astype(int)
#     return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]

def center_crop(image):
    t, l = (np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE) // 2
    return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]
