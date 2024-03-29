#!/usr/bin/env python3
""" Evaluator """

import sys
import tensorflow as tf
from utils import decode_img, image_patches, write_tensor_as_image
from model import image_diff, UPSCALER_FACTOR

def main():
    """ Main function """
    try:
        image_path = sys.argv[1]
    except:
        print("Usage: {} <image path>".format(sys.argv[0]))
        exit(-1)

    try:
        model_path = sys.argv[2]
    except:
        model_path = './saved_model'

    PATCH_SIZE = 240 // UPSCALER_FACTOR
    N_CHANNELS = 3

    model = tf.keras.models.load_model(model_path)

    image = decode_img(image_path, N_CHANNELS)
    patches = image_patches(image, PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    model_out = model(patches)
    for idx, (patch_in, patch_out) in enumerate(zip(patches, model_out)):
        write_tensor_as_image("{}a.png".format(idx), patch_in)
        write_tensor_as_image("{}b.png".format(idx), patch_out)
    #image = tf.expand_dims(image, axis=0)
    #model_out = model(image)
    #model_out = tf.squeeze(model_out, axis=0)
    #write_tensor_as_image("out.png", model_out)

if __name__ == "__main__":
    main()
