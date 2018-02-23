import pickle

import numpy as np
import tensorflow as tf

from utils import model
from utils.io_utils import load_vgg19, load_image, save_image, get_filepaths_in_dir, get_image_iterator
from utils.model import swap

photo_file_paths = sorted(get_filepaths_in_dir('patch_based_fast_nst/origin_image'))
photo = load_image(photo_file_paths[0])
photo = photo.reshape([1, *photo.shape])

image_height = photo.shape[1]
image_width = photo.shape[2]
channel = 3
batch = 1
patch_h = 3
patch_w = 3
means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
h_shape = [batch, int(image_height / 4), int(image_width / 4), 256]
vgg = load_vgg19('pretrained_model/imagenet-vgg-verydeep-19.mat')
swap_photo_input = tf.Variable(tf.zeros([batch, image_height, image_width, channel], tf.float32))
swap_ph_sub_mean = tf.subtract(swap_photo_input, means)
_, _, _, _, swap_relu3_1, _, _ = \
    model.build_vgg19(vgg, image_height, image_width, channel, batch, input_tensor=swap_ph_sub_mean)

_, H_h, H_w, H_c = swap_relu3_1.get_shape().as_list()

style_h_now = pickle.load(open('patch_based_fast_nst/style_h_file/style_h.pickle', 'rb'))
style_placeholder = tf.placeholder(tf.float32, style_h_now.shape)

style_patchs = tf.extract_image_patches(style_placeholder, ksizes=[1, patch_h, patch_w, 1], strides=[1, 1, 1, 1],
                                        rates=[1, 1, 1, 1], padding='VALID')

normalized_style_patchs = tf.nn.l2_normalize(style_patchs, 3)
normalized_style_patch_filters = tf.transpose(tf.reshape(normalized_style_patchs,
                                                         shape=[normalized_style_patchs.shape[1] *
                                                                normalized_style_patchs.shape[2], patch_h,
                                                                patch_w, -1]), perm=[1, 2, 3, 0])

unnormalized_style_patch_filters = tf.reshape(style_patchs,
                                              shape=[normalized_style_patchs.shape[1] *
                                                     normalized_style_patchs.shape[2], patch_h,
                                                     patch_w, -1])

conv_K = tf.nn.conv2d(swap_relu3_1, filter=normalized_style_patch_filters, strides=[1, 1, 1, 1], padding='VALID')

conv_K_argmax = tf.argmax(conv_K, 3)

phi_cs_placeholder = tf.placeholder(tf.float32, h_shape)

lambd = 1e-3
tv_loss = lambd * tf.reduce_sum(tf.image.total_variation(swap_photo_input))
content_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(swap_relu3_1, phi_cs_placeholder))))
total_loss = content_loss + tv_loss
train_step = tf.train.AdamOptimizer(2).minimize(total_loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

photo_ir = get_image_iterator(photo_file_paths, batch=batch, shuffle=False, do_resize=False)
sess.run(photo_ir.initializer)

for i in range(len(photo_file_paths)):
    photo = sess.run(photo_ir.get_next())

    noise_ratio = 0.2
    generated_image = noise_ratio * np.random.uniform(-20, 20, photo.shape) + (1 - noise_ratio) * photo.copy()

    sess.run(swap_photo_input.assign(photo))
    photo_h_now = sess.run(swap_relu3_1)
    conv_K_argmax_ps, uspf_ps = sess.run([conv_K_argmax, unnormalized_style_patch_filters],
                                         feed_dict={style_placeholder: style_h_now})

    conv_K_argmax_ps = conv_K_argmax_ps[0]
    phi_cs = swap([H_h, H_w, H_c], conv_K_argmax_ps, uspf_ps)
    phi_cs = phi_cs.reshape(1, *phi_cs.shape)
    sess.run(swap_photo_input.assign(generated_image))
    for k in range(100):
        sess.run(train_step, feed_dict={phi_cs_placeholder: phi_cs})
        print(i, k, sess.run(total_loss, feed_dict={phi_cs_placeholder: phi_cs}))

    generated_image = sess.run(swap_photo_input)
    generated_image = generated_image.reshape(generated_image.shape[1:])

    generated_image = np.clip(generated_image, 0, 255).astype('uint8')
    save_image(generated_image, 'patch_based_fast_nst/output_image/%04d.jpg' % (i,))
    print('output image %04d.jpg' % (i,))
