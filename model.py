from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf 

def image_similarity(y_true, y_pred):
    return 1.0 - tf.math.reduce_mean(tf.math.abs(y_pred-y_true))

# Helper
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def upscaler(input_size, n_channels):
    inp = layers.Input(shape=(input_size, input_size, n_channels))

    stage1 = layers.Conv2DTranspose(n_channels * 128, (1, 1), strides=(1, 1), padding='same')(inp)
    state1 = layers.BatchNormalization()(stage1)
    state1 = layers.LeakyReLU()(stage1)

    stage2 = layers.Conv2DTranspose(n_channels * 64, (5, 5), strides=(2, 2), padding='same')(stage1)
    state2 = layers.BatchNormalization()(stage2)
    state2 = layers.LeakyReLU()(stage2)

    stage3 = layers.Conv2DTranspose(n_channels, (5, 5), strides=(2, 2), padding='same')(stage2)

    return models.Model(inputs=inp, outputs=stage3)
UPSCALER_FACTOR = 4

@tf.function
def upscaler_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator(input_size, n_channels):
    inp = layers.Input(shape=(input_size, input_size, n_channels))

    conv0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inp)

    conv0 = layers.LeakyReLU()(conv0)
    conv0 = layers.Dropout(0.3)(conv0)

    conv1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(conv0)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.Dropout(0.3)(conv1)

    flat = layers.Flatten()(conv1)
    boolean = layers.Dense(1)(flat)

    return models.Model(inputs=inp, outputs=boolean)

@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

class Trainer:
    def __init__(self, image_size, n_channels):
        """
        image_size: Full-scale image size (Must be divisible by scale_factor)
        scale_factor: Scale factor from input to output (Must be divisible by 2)
        """
        self.image_size = image_size
        downscaled_size = image_size // UPSCALER_FACTOR

        self.upscaler = upscaler(downscaled_size, n_channels)
        self.discriminator = discriminator(image_size, n_channels)

        self.upscaler_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.downscaled_size_tensor = tf.constant([downscaled_size, downscaled_size])

    @tf.function
    def train_step(self, image_a, image_b):
        """
        Downscales A, upscales that result synthetically to the original size,
        then the success of that is compared to B.
        """

        image_a_downscale = tf.image.resize(image_a,
                               size=self.downscaled_size_tensor,
                               method='bilinear')

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            upscaled_a = self.upscaler(image_a_downscale, training=True)
    
            real_output = self.discriminator(image_b, training=True)
            fake_output = self.discriminator(upscaled_a, training=True)
    
            gen_loss = upscaler_loss(fake_output)
            #gen_loss += cross_entropy(image_a, upscaled_a) # Incentivise making the image similar to the original
            disc_loss = discriminator_loss(real_output, fake_output)
    
        gradients_of_upscaler = gen_tape.gradient(gen_loss, self.upscaler.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    
        self.upscaler_optimizer.apply_gradients(zip(gradients_of_upscaler, self.upscaler.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return (gen_loss, disc_loss)
