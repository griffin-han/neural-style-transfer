import os

import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf


def load_vgg19(vgg19_path):
    return scipy.io.loadmat(vgg19_path)


def save_image(image_content, image_path):
    scipy.misc.imsave(image_path, image_content)


def load_image(image_path, new_hw_tuple=None, max_len_of_height_widith=None):
    image = scipy.misc.imread(image_path)
    if new_hw_tuple:
        image = scipy.misc.imresize(image, new_hw_tuple)
    if not max_len_of_height_widith or np.max(image.shape) < max_len_of_height_widith:
        return image
    else:
        resized_factor = 1.0 * max_len_of_height_widith / np.max(image.shape)
        resized_height = int(resized_factor * image.shape[0])
        resize_width = int(resized_factor * image.shape[1])
        return scipy.misc.imresize(image, (resized_height, resize_width))


def get_image_iterator(image_file_paths, resize_tuple=(256, 256), batch=2, repeat=None, shuffle=True, do_resize=True):
    """
    use after
    sess = tf.InteractiveSession()
    sess.run(iterator.initializer)
    
    get value by 
    sess.run(iterator.get_next())
    """

    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        if do_resize:
            image_resized = tf.image.resize_images(image_decoded, resize_tuple)
        else:
            image_resized = image_decoded
        return image_resized

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_file_paths),))
    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(repeat)
    iterator = dataset.make_initializable_iterator()
    return iterator


def get_filepaths_in_dir(dir_path):
    ret = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            ret.append(os.path.join(root, f))
    return ret
