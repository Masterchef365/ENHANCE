#!/usr/bin/env python3
""" Evaluator """

import sys
import tensorflow as tf
from utils import decode_img
from model import image_similarity


def main():
    """ Main function """
    try:
        image_path = sys.argv[1]
    except:
        print("Usage: {} <image path>".format(sys.argv[0]))
        exit(-1)

    model = tf.keras.models.load_model('./saved_model')

    image = decode_img(image_path, 1)
    model_out = model(tf.expand_dims(image, axis=0))
    img = tf.squeeze(model_out, axis=0)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=False)
    img = tf.image.encode_png(img)
    tf.io.write_file("out.png", img)

    #plt.figure()
    #plt.imshow(tf.squeeze(image))
    #plt.imshow(tf.squeeze(model_out))
    #plt.show()


if __name__ == "__main__":
    main()
