from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf 

@tf.function
def image_diff(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.abs(y_pred-y_true))

#import numpy as np
#a = np.random.random((1, 33, 33, 1))
#b = np.random.random((1, 33, 33, 1))
#print(image_diff(a, b))
#print(image_diff(a, a))
#print(image_diff(b, b))

# Helper
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def upscaler_unit(input_layer, units, size, stride, bnlrelu=True):
    layer = layers.Conv2DTranspose(units, (size, size), strides=(stride, stride), padding='same')(input_layer)
    if bnlrelu:
        layer = layers.BatchNormalization()(layer)
        layer = layers.LeakyReLU()(layer)
    return layer

def upscaler(input_size, n_channels):
    inp = layers.Input(shape=(input_size, input_size, n_channels))
    stage0 = upscaler_unit(inp, n_channels * 128, 5, 1)
    stage1 = upscaler_unit(stage0, n_channels * 64, 5, 2)
    stage2 = upscaler_unit(stage1, n_channels, 5, 2, False)
    return models.Model(inputs=inp, outputs=stage2)
UPSCALER_FACTOR = 4

@tf.function
def upscaler_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_unit(input_layer, units, size, stride):
    layer = layers.Conv2D(units, (size, size), strides=(stride, stride), padding='same')(input_layer)
    layer = layers.LeakyReLU()(layer)
    layer = layers.Dropout(0.3)(layer)
    return layer

def discriminator(input_size, n_channels):
    inp = layers.Input(shape=(input_size, input_size, n_channels))
    stage0 = discriminator_unit(inp, 64, 5, 2)
    stage1 = discriminator_unit(stage0, 128, 5, 2)
    stage2 = discriminator_unit(stage1, 256, 5, 2)
    flat = layers.Flatten()(stage2)
    boolean = layers.Dense(1)(flat)
    return models.Model(inputs=inp, outputs=boolean)

@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

class Trainer:
    def __init__(self, image_size, n_channels, similarize_factor, learning_rate):
        """
        image_size: Full-scale image size (Must be divisible by scale_factor)
        scale_factor: Scale factor from input to output (Must be divisible by 2)
        """
        self.similarize_factor = tf.constant(similarize_factor)
        self.image_size = image_size
        downscaled_size = image_size // UPSCALER_FACTOR

        self.upscaler = upscaler(downscaled_size, n_channels)
        self.discriminator = discriminator(image_size, n_channels)

        self.upscaler_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.downscaled_size_tensor = tf.constant([downscaled_size, downscaled_size])

    @tf.function
    def train_step(self, image_a):
        """
        Downscales A, upscales that result synthetically to the original size,
        then the success of that is compared to B.
        """

        image_a_downscale = tf.image.resize(image_a,
                               size=self.downscaled_size_tensor,
                               method='bilinear')

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            upscaled_a = self.upscaler(image_a_downscale, training=True)
    
            real_output = self.discriminator(image_a, training=True)
            fake_output = self.discriminator(upscaled_a, training=True)
    
            gen_loss = upscaler_loss(fake_output)
            img_diff = image_diff(image_a, upscaled_a)
            gen_loss += self.similarize_factor * img_diff
            disc_loss = discriminator_loss(real_output, fake_output)
    
        gradients_of_upscaler = gen_tape.gradient(gen_loss, self.upscaler.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    
        self.upscaler_optimizer.apply_gradients(zip(gradients_of_upscaler, self.upscaler.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return (gen_loss, disc_loss, img_diff)
