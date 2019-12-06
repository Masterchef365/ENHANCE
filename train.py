#!/usr/bin/env python3
""" Trainer """

import tensorflow as tf
import datetime
from model import get_model
from utils import dataset_patches
import numpy as np
from tensorflow import keras
import sys

def resized_dataset(patch_size, patch_downscaled, stride, n_channels, data_dir):
    """
    Create the dataset
    """
    patches_ds = dataset_patches(data_dir + "/*", patch_size, stride, n_channels)\
        .unbatch().shuffle(buffer_size=4211).batch(1)

    def resize_op(patch):
        return tf.image.resize(patch,
                               size=[patch_downscaled, patch_downscaled],
                               method='bilinear')

    patches_ds = patches_ds.prefetch(tf.data.experimental.AUTOTUNE)
    patches_resized = patches_ds.map(lambda patch: (resize_op(patch), patch))
    patches_resized = patches_resized.prefetch(tf.data.experimental.AUTOTUNE)

    return patches_resized


DOWNSCALED_SIZE = 48
UPSCALE_FACTOR = 4

def main():
    try:
        dataset_dir = sys.argv[1]
    except:
        print("Usage: {} <training data>".format(sys.argv[0]))
        exit(-1)

    model = get_model(DOWNSCALED_SIZE, 1)

    model.summary()

    dataset = resized_dataset(DOWNSCALED_SIZE * UPSCALE_FACTOR, DOWNSCALED_SIZE, 40, 1, dataset_dir)

    model.fit(dataset, epochs=3)

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.save('saved_model', save_format='tf')

if __name__ == "__main__":
    main()

