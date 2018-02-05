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

tf.reset_default_graph()

swap_ph = tf.placeholder(tf.float32, shape=[batch, image_height, image_width, channel])
swap_ph_sub_mean = tf.subtract(swap_ph, means)
_, _, _, _, swap_relu3_1, _, _ = \
    model.build_vgg19(vgg, image_height, image_width, channel, batch, input_tensor=swap_ph_sub_mean)

_, H_h, H_w, H_c = swap_relu3_1.get_shape().as_list()

photo_placeholder = tf.placeholder(tf.float32, [1, H_h, H_w, H_c])

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
conv_K = tf.nn.conv2d(photo_placeholder, filter=normalized_style_patch_filters, strides=[1, 1, 1, 1], padding='VALID')

conv_K_argmax = tf.argmax(conv_K, 3)

h, f_h = model.build_inverse_part_vgg19_network(
    tf.placeholder(tf.float32, shape=[batch, int(image_height / 4), int(image_width / 4), 256]))

_, _, _, _, phi_f_h, _, _ = model.build_vgg19(vgg, *h_shape[1:], batch, f_h)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'patch_based_fast_nst/model_files/model.ckpt')

photo_ir = get_image_iterator(photo_file_paths, batch=batch, shuffle=False, do_resize=False)
sess.run(photo_ir.initializer)

for i in range(len(photo_file_paths)):
    photo = sess.run(photo_ir.get_next())
    photo_h_now = sess.run(swap_relu3_1, feed_dict={swap_ph: photo})
    conv_K_argmax_ps, uspf_ps = sess.run([conv_K_argmax, unnormalized_style_patch_filters],
                                         feed_dict={photo_placeholder: photo_h_now,
                                                    style_placeholder: style_h_now})

    conv_K_argmax_ps = conv_K_argmax_ps[0]
    styled_h = swap([H_h, H_w, H_c], conv_K_argmax_ps, uspf_ps)
    styled_h = styled_h.reshape(1, *styled_h.shape)
    f_h_val = sess.run(f_h, feed_dict={h: styled_h})
    f_h_val += means

    f_h_val = np.clip(f_h_val, 0, 255).astype('uint8')
    f_h_val = f_h_val.reshape(f_h_val.shape[1:])
    save_image(f_h_val, 'patch_based_fast_nst/output_image/%04d.jpg' % (i,))
    print('output image %04d.jpg' % (i,))
