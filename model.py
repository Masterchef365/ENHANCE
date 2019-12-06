from tensorflow import keras
import tensorflow as tf 

def image_similarity(y_true, y_pred):
    return 1.0 - tf.math.reduce_mean(tf.math.abs(y_pred-y_true))

def get_model(input_size, n_channels):
    inp = keras.layers.Input(shape=(input_size, input_size, n_channels))

    stage1 = keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(inp)
    state1 = keras.layers.BatchNormalization()(stage1)
    state1 = keras.layers.LeakyReLU()(stage1)

    stage2 = keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(stage1)
    state2 = keras.layers.BatchNormalization()(stage2)
    state2 = keras.layers.LeakyReLU()(stage2)

    model = keras.models.Model(inputs=inp, outputs=stage2)

    model.compile(optimizer='adam',
              loss='binary_crossentropy')

    return model

    #conv_layers = keras.layers.Conv2D(1, (9, 9), padding='same')(inp)
    #combine = stage1 + conv_layers
    #stage2 = keras.layers.Conv2DTranspose(1, (3, 3), strides=(3, 3), padding='same')(combine)
    #model = keras.models.Model(inputs=inp, outputs=combine)


if __name__ == "__main__":
    import numpy as np
    a = np.random.random((35, 35, 1))
    b = np.random.random((35, 35, 1))
    print(image_similarity(a, b))
    print(image_similarity(a, a))
