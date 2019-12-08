import tensorflow as tf

def write_tensor_as_image(path, tensor):
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint16, saturate=True)
    img = tf.image.encode_png(tensor)
    tf.io.write_file(path, img)


def decode_img(file_path, n_channels):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=n_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def image_patches(image, size, stride, n_channels):
    patch_size = [1, size, size, 1]
    patch_stride = [1, stride, stride, 1]
    patches = tf.image.extract_patches([image], patch_size, patch_stride, [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, shape=[-1, size, size, n_channels])
    return patches


def dataset_patches(directory, patch_size, stride, n_channels):
    files = tf.data.Dataset.list_files(directory)
    return files\
            .map(lambda fname: decode_img(fname, n_channels))\
            .map(lambda img: image_patches(img, patch_size, stride, n_channels))
