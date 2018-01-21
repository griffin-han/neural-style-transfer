import numpy as np
import tensorflow as tf

import model as model
from io_utils import load_image, load_vgg19, save_image
from misc import generate_blank_image

# the mean value to adjust, to match the vgg19 input scale
means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
# load content image, style image and init the generated image from a blank image
content_image = load_image('origin_image/content.jpg', max_len_of_height_widith=600) - means
height, width = content_image.shape[0], content_image.shape[1]
style_image = load_image('style_image/style.jpg', new_hw_tuple=(height, width)) - means
generated_image = generate_blank_image(height, width)

# reshape the three images to the same size
shape = [1, height, width, 3]
content_image = np.reshape(content_image, shape)
style_image = np.reshape(style_image, shape)
generated_image = np.reshape(generated_image, shape)

# load the vgg19 model
vgg = load_vgg19('pretrained_model/imagenet-vgg-verydeep-19.mat')
sess = tf.InteractiveSession()
network_input, conv4_2, relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 = model.build_network(vgg, height, width)

# compute the content loss
sess.run(network_input.assign(content_image))
content_out = sess.run(conv4_2)
g_out = conv4_2
content_loss = model.get_content_loss(g_out, content_out)

# compute the style loss
sess.run(network_input.assign(style_image))
s1_1, s2_1, s3_1, s4_1, s5_1 = sess.run([relu1_1, relu2_1, relu3_1, relu4_1, relu5_1])
style_loss = model.get_style_loss(s1_1, s2_1, s3_1, s4_1, s5_1, relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)

# compute the total loss
total_loss = model.get_total_loss(content_loss, style_loss, alpha=3e-4)

# use adam to optimize
step = tf.train.AdamOptimizer(1).minimize(total_loss)

# assign the init image to the vgg19 model
sess.run(tf.global_variables_initializer())
sess.run(network_input.assign(generated_image))

# optimize the total loss, output the result image for each 100 iteration
for i in range(6000):
    sess.run(step)
    train_step_total_loss = sess.run([total_loss, content_loss, style_loss])
    print('iter {0}: total_loss={1}, content_loss={2}, style_loss={3}'.format(i, *train_step_total_loss))
    if i % 100 == 0:
        tmp_gen_img = sess.run(network_input)
        tmp_gen_img = np.reshape(tmp_gen_img, [height, width, 3])
        tmp_gen_img = np.clip(tmp_gen_img + means, 0, 255).astype('uint8')
        save_image(tmp_gen_img, 'output_image/{id}.jpg'.format(id=i))
save_image(tmp_gen_img, 'output_image/{id}.jpg'.format(id=i))
