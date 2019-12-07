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

    PATCH_SIZE = 192
    N_CHANNELS = 1

    # Set up trainer
    trainer = Trainer(PATCH_SIZE, N_CHANNELS)
    trainer.discriminator.summary()
    trainer.upscaler.summary()

    # Prepare datasets
    patches_ds = dataset_patches(dataset_dir + "/*", PATCH_SIZE, PATCH_SIZE, N_CHANNELS).unbatch()

    d_a = patches_ds.shuffle(buffer_size=1000).batch(40)
    d_b = patches_ds.shuffle(buffer_size=1000).batch(40)
    dataset = tf.data.Dataset.zip((d_a, d_b))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Train
    N_EPOCHS = 10
    SAVE_FREQUENCY = 1000
    count = 0
    for epoch in range(N_EPOCHS):
        tf.print("\n%%%%% Epoch {} %%%%%".format(epoch))
        for idx, tup in enumerate(dataset):

            a, b = tup
            gen_loss, disc_loss = trainer.train_step(a, b)

            if idx % SAVE_FREQUENCY == 0:
                tf.print("\nSaving... (Epoch {})".format(epoch))
                trainer.upscaler.save('saved_model', save_format='tf')

            tf.print("  {:4}/{:4} | Gen: {:0.3f} Dist: {:0.3f}".format(idx, count, gen_loss, disc_loss), end='\r')

        count = idx

    print("\nFinished training")

if __name__ == "__main__":
    main()

