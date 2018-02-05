import numpy as np
import tensorflow as tf

from utils import model
from utils.io_utils import load_vgg19, get_image_iterator, get_filepaths_in_dir

image_height = 256
image_width = 256
channel = 3
batch = 32
half_batch = 4
patch_h = 3
patch_w = 3
means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
h_shape = [batch, int(image_height / 4), int(image_width / 4), 256]
vgg = load_vgg19('pretrained_model/imagenet-vgg-verydeep-19.mat')

tf.reset_default_graph()

swap_ph = tf.placeholder(tf.float32, shape=[batch, image_height, image_width, channel])
swap_ph_sub_mean = tf.subtract(swap_ph, means)
# the network to compute H value
_, _, _, _, swap_relu3_1, _, _ = \
    model.build_vgg19(vgg, image_height, image_width, channel, batch, input_tensor=swap_ph_sub_mean)

# build inverse net followed by vgg19: h->f_h->phi_f_h
h, f_h = model.build_inverse_part_vgg19_network(
    tf.placeholder(tf.float32, shape=[batch, int(image_height / 4), int(image_width / 4), 256]))

_, _, _, _, phi_f_h, _, _ = model.build_vgg19(vgg, *h_shape[1:], batch, f_h)
total_loss = model.get_style_swap_loss(phi_f_h, h, f_h)

train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

photo_file_paths = get_filepaths_in_dir('patch_based_fast_nst/mscoco2014_images')
photo_ir = get_image_iterator(photo_file_paths, batch=batch)

sess.run([photo_ir.initializer])

for i in range(10000):
    photos = sess.run(photo_ir.get_next())
    if photos.shape[0] != batch:
        photos = sess.run(photo_ir.get_next())
    h_now = sess.run(swap_relu3_1, feed_dict={swap_ph: photos})
    sess.run(train_step, feed_dict={h: h_now})
    tmp_loss = sess.run(total_loss, feed_dict={h: h_now})
    print(i, tmp_loss)
saver = tf.train.Saver()
saver.save(sess, 'patch_based_fast_nst/model_files/model.ckpt')
