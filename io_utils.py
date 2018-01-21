import numpy as np
import scipy.io
import scipy.misc


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
