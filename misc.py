import numpy as np


def generate_blank_image(height, width, channel=3):
    return np.zeros([height, width, channel])
