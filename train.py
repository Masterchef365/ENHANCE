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

def main():
    try:
        dataset_dir = sys.argv[1]
    except:
        print("Usage: {} <training data>".format(sys.argv[0]))
        exit(-1)

    N_EPOCHS = 10
    SAVE_FREQUENCY = 1000

    PATCH_SIZE = 160
    N_CHANNELS = 3

    BATCH_SIZE = 20

    UPSCALER_CKPT_DIR = 'checkpoint/upscaler_saved_model'
    DISCRIMINATOR_CKPT_DIR = 'checkpoint/discriminator_saved_model'
    MODEL_SAVE_DIR = 'saved_model'

    # Set up trainer
    trainer = Trainer(PATCH_SIZE, N_CHANNELS)

    try:
        trainer.upscaler.load_weights(UPSCALER_CKPT_DIR)
        trainer.discriminator.load_weights(DISCRIMINATOR_CKPT_DIR)
        print("Loaded weights from file")
    except:
        print("Failed to find weights. Starting from scratch.")

    trainer.discriminator.summary()
    trainer.upscaler.summary()

    # Prepare datasets
    patches_ds = dataset_patches(dataset_dir + "/*", PATCH_SIZE, PATCH_SIZE, N_CHANNELS).unbatch()

    d_a = patches_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    d_b = patches_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    dataset = tf.data.Dataset.zip((d_a, d_b))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Train
    count = 0
    for epoch in range(N_EPOCHS):
        tf.print("\n%%%%% Epoch {} %%%%%".format(epoch))
        for batch_idx, tup in enumerate(dataset):

            a, b = tup
            gen_loss, disc_loss = trainer.train_step(a, b)

            if batch_idx % SAVE_FREQUENCY == 0:
                tf.print("\nSaving... (Epoch {})".format(epoch))
                trainer.upscaler.save_weights(UPSCALER_CKPT_DIR, save_format='tf')
                trainer.discriminator.save_weights(DISCRIMINATOR_CKPT_DIR, save_format='tf')
                trainer.upscaler.save('saved_model', save_format='tf')

            tf.print("  {:4}/{:4} | Gen: {:0.3f} Disc: {:0.3f}".format(batch_idx * BATCH_SIZE, count * BATCH_SIZE, gen_loss, disc_loss), end='\r')

        count = batch_idx

    print("\nFinished training")

if __name__ == "__main__":
    main()

