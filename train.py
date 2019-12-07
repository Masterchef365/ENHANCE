#!/usr/bin/env python3
""" Trainer """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import datetime
from model import Trainer
from utils import dataset_patches
import numpy as np
from tensorflow import keras
import sys

#from model import discriminator, upscaler
#disc = discriminator(PATCH_SIZE)
#upsc = upscaler(PATCH_SIZE)
#a = np.random.random((8, PATCH_SIZE, PATCH_SIZE, 1))
#print(disc(a))
#print(upsc(a))
#exit(-1)

def main():
    try:
        dataset_dir = sys.argv[1]
    except:
        print("Usage: {} <training data>".format(sys.argv[0]))
        exit(-1)

    PATCH_SIZE = 96
    N_CHANNELS = 3

    trainer = Trainer(PATCH_SIZE, N_CHANNELS)
    trainer.discriminator.summary()

    patches_ds = dataset_patches(dataset_dir + "/*", PATCH_SIZE, PATCH_SIZE, N_CHANNELS).unbatch()

    d_a = patches_ds.shuffle(buffer_size=1000).batch(40)
    d_b = patches_ds.shuffle(buffer_size=1000).batch(40)
    dataset = tf.data.Dataset.zip((d_a, d_b))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for epoch in range(10):
        print("%%%%% Epoch {} %%%%%".format(epoch))
        for tup in dataset:
            a, b = tup
            tf.print(trainer.train_step(a, b))

    print("Finished training")

    #log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    trainer.upscaler.save('saved_model', save_format='tf')

if __name__ == "__main__":
    main()

