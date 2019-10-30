#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

#model = keras.Sequential([
#    keras.layers.Conv2D(1, (3, 3), strides=(1, 1)),
#    keras.layers.Dense(1, activation='relu'),
#    keras.layers.Conv2D(1, (3, 3), strides=(1, 1)),
#])
#
#model.compile(optimizer='adam',
#              loss='mean_squared_error',
#              metrics=['accuracy'])
#
#image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#train_data_gen = image_generator.flow_from_directory(directory=str("./data"),
#                                                     shuffle=True,
#                                                     target_size=(None, None))
#
#files = tf.data.Dataset.list_files(str(data_dir/'*/*'))
#
#model.fit(x=image_generator, y=image_generator)

def decode_img(file_path):
    image = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    return image

def image_patches(image, size):
    patch_size = [1,size,size,1]
    patches = tf.image.extract_patches([image], patch_size, patch_size, [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, shape=[-1, size, size, 3])
    return patches

def dataset_patches(directory, patch_width):
    files = tf.data.Dataset.list_files(directory)
    return files.map(decode_img).map(lambda img: image_patches(img, patch_width)).unbatch()

patch_size = 90
patch_downscaled = 35
dataset_patches = dataset_patches("./data/*", 90)
patches_sidebyside = dataset_patches.map(lambda patch: (patch, tf.image.resize(patch, size=[patch_downscaled, patch_downscaled], method='nearest')))
