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

PATCH_SIZE = 48

EPOCHS = 1

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

    trainer = Trainer(PATCH_SIZE)
    trainer.discriminator.summary()

    patches_ds = iter(dataset_patches(dataset_dir + "/*", PATCH_SIZE, PATCH_SIZE, 1))

    for _ in range(EPOCHS):
        try:
            last = next(patches_ds)
            while True:
                current = next(patches_ds)
                gen_loss, disc_loss = trainer.train_step(last, current)
                print("{} {}".format(gen_loss, disc_loss))
                last = current
        except StopIteration:
            pass

    #model.summary()

    #dataset = resized_dataset(PATCH_SIZE * UPSCALE_FACTOR, PATCH_SIZE, 40, 1, dataset_dir)

    #model.fit(dataset, epochs=3)

    #log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model.save('saved_model', save_format='tf')

if __name__ == "__main__":
    main()

