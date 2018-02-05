import pickle

import numpy as np
import tensorflow as tf

from utils import model
from utils.io_utils import load_vgg19, load_image

style = load_image('patch_based_fast_nst/style_image/style.jpg')
style = style.reshape([1, *style.shape])

image_height = style.shape[1]
image_width = style.shape[2]
channel = 3
batch = 1
patch_h = 3
patch_w = 3
means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
h_shape = [batch, int(image_height / 4), int(image_width / 4), 256]
vgg = load_vgg19('pretrained_model/imagenet-vgg-verydeep-19.mat')

tf.reset_default_graph()

swap_ph = tf.placeholder(tf.float32, shape=[batch, image_height, image_width, channel])
swap_ph_sub_mean = tf.subtract(swap_ph, means)
_, _, _, _, swap_relu3_1, _, _ = \
    model.build_vgg19(vgg, image_height, image_width, channel, batch, input_tensor=swap_ph_sub_mean)

h, f_h = model.build_inverse_part_vgg19_network(
    tf.placeholder(tf.float32, shape=[batch, int(image_height / 4), int(image_width / 4), 256]))

_, _, _, _, phi_f_h, _, _ = model.build_vgg19(vgg, *h_shape[1:], batch, f_h)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

style_h_now = sess.run(swap_relu3_1, feed_dict={swap_ph: style})

pickle.dump(style_h_now, open('patch_based_fast_nst/style_h_file/style_h.pickle', mode='wb'))
