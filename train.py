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


@tf.function
def train_epoch(trainer, dataset):
    for batch_idx, tup in enumerate(dataset):
        a, b = tup
        gen_loss, disc_loss = trainer.train_step(a, b)
        tf.print(gen_loss, disc_loss)


def main():
    try:
        dataset_dir = sys.argv[1]
    except:
        print("Usage: {} <training data>".format(sys.argv[0]))
        exit(-1)

    N_EPOCHS = 50
    #SAVE_FREQUENCY = 1000

    PATCH_SIZE = 240
    N_CHANNELS = 3

    BATCH_SIZE = 10

    SIMILARIZE_FACTOR = 0.1

    UPSCALER_CKPT_DIR = 'checkpoint/upscaler_saved_model'
    DISCRIMINATOR_CKPT_DIR = 'checkpoint/discriminator_saved_model'
    MODEL_SAVE_DIR = 'saved_model'

    # Set up trainer
    trainer = Trainer(PATCH_SIZE, N_CHANNELS, SIMILARIZE_FACTOR)

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
    for epoch in range(N_EPOCHS):
        print("\n%%%%% Epoch {} %%%%%".format(epoch))
        train_epoch(trainer, dataset)

        print("\nSaving...")
        trainer.upscaler.save_weights(UPSCALER_CKPT_DIR, save_format='tf')
        trainer.discriminator.save_weights(DISCRIMINATOR_CKPT_DIR, save_format='tf')
        trainer.upscaler.save(MODEL_SAVE_DIR, save_format='tf')

    print("\nFinished training")

if __name__ == "__main__":
    main()

