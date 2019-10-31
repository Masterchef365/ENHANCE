#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import datetime

def get_model():
    model = keras.Sequential([
        keras.layers.Conv2DTranspose(1, (3, 3), strides=(3, 3)),
    ])

    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

    return model

def decode_img(file_path, n_channels):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=n_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    return image

def image_patches(image, size, n_channels):
    patch_size = [1,size,size,1]
    patches = tf.image.extract_patches([image], patch_size, patch_size, [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, shape=[-1, size, size, n_channels])
    return patches

def dataset_patches(directory, patch_size, n_channels):
    files = tf.data.Dataset.list_files(directory)
    return files.map(lambda fname: decode_img(fname, n_channels)).map(lambda img: image_patches(img, patch_size, n_channels))

def my_dataset(patch_size, patch_downscaled, n_channels, data_dir):
    patches_ds = dataset_patches(data_dir + "/*", patch_size, n_channels).unbatch().shuffle(buffer_size=4000).batch(1)
    patches_resized = patches_ds.map(lambda patch: (tf.image.resize(patch, size=[patch_downscaled, patch_downscaled], method='nearest'), patch))

    return patches_resized


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_dir = "./tf_ckpts/"

model = get_model()

try:
    latest = tf.train.latest_checkpoint(ckpt_dir)
    model.load_weights(latest)
    print("Loaded checkpoint from {}.".format(latest))
except:
    print("Could not find checkpoint at {}. Continuing from scratch.".format(ckpt_dir))

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir + "/ckpt", save_weights_only=True, verbose=1, save_freq='epoch')

dataset = my_dataset(99, 33, 1, "./data")
model.fit(dataset, 
          callbacks=[
              #tensorboard_callback,
              cp_callback,
              ],
          epochs=1)
