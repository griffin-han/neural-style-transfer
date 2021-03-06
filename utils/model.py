import numpy as np
import tensorflow as tf


def build_inverse_part_vgg19_network(input_tensor, e=1e-8):
    """
    shape from relu3_1 to origin image
    :param input_tensor: 
    :param e: 
    :return: 
    """

    def _instance_norm(x):
        m, v = tf.nn.moments(x, [1, 2], keep_dims=True)

        return tf.div(tf.subtract(x, m), tf.sqrt(tf.add(v, e)))

    _, height, width, channel = input_tensor.get_shape().as_list()

    filter_conv1 = tf.Variable(tf.random_normal([3, 3, 256, 128]))
    conv1 = tf.nn.conv2d(input_tensor, filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
    inst_norm1 = _instance_norm(conv1)
    relu1 = tf.nn.relu(inst_norm1)
    _, h1, w1, c1 = relu1.get_shape().as_list()
    nnu1 = tf.image.resize_images(relu1, [h1 * 2, w1 * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    filter_conv2 = tf.Variable(tf.random_normal([3, 3, 128, 128]))
    conv2 = tf.nn.conv2d(nnu1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME')
    inst_norm2 = _instance_norm(conv2)
    relu2 = tf.nn.relu(inst_norm2)

    filter_conv3 = tf.Variable(tf.random_normal([3, 3, 128, 64]))
    conv3 = tf.nn.conv2d(relu2, filter_conv3, strides=[1, 1, 1, 1], padding='SAME')
    inst_norm3 = _instance_norm(conv3)
    relu3 = tf.nn.relu(inst_norm3)
    _, h3, w3, c3 = relu3.get_shape().as_list()

    nnu2 = tf.image.resize_images(relu3, [h3 * 2, w3 * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    filter_conv4 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
    conv4 = tf.nn.conv2d(nnu2, filter_conv4, strides=[1, 1, 1, 1], padding='SAME')
    inst_norm4 = _instance_norm(conv4)
    relu4 = tf.nn.relu(inst_norm4)

    filter_conv5 = tf.Variable(tf.random_normal([3, 3, 64, 3]))
    conv5 = tf.nn.conv2d(relu4, filter_conv5, strides=[1, 1, 1, 1], padding='SAME')

    return input_tensor, conv5


def get_style_swap_loss(phi_f_h, h, f_h, lambd=1e-6):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(phi_f_h, h)), axis=[1, 2, 3],
                                                keep_dims=False)) + lambd * tf.image.total_variation(f_h))


######################


def build_vgg19(vgg19_model, height, width, channel=3, batch=1, input_tensor=None):
    def _weights(layer):
        w, b = vgg19_model['layers'][0][layer][0][0][2][0]
        return w, b

    def _build_conv2d_and_relu(pre_layer_data, this_layer_num):
        w, b = _weights(this_layer_num)
        w = np.transpose(w, (1, 0, 2, 3))
        w = tf.constant(w)

        b = tf.constant(np.reshape(b, [b.size]))
        conv = tf.nn.conv2d(pre_layer_data, w, strides=[1, 1, 1, 1], padding='SAME')
        return conv, tf.nn.relu(tf.add(conv, b))

    def _build_pooling(pre_layer):
        return tf.nn.avg_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    if input_tensor is not None:
        network_input = input_tensor
    else:
        network_input = tf.Variable(tf.zeros((batch, height, width, channel), dtype=tf.float32))
    conv1_1, relu1_1 = _build_conv2d_and_relu(network_input, 0)
    conv1_2, relu1_2 = _build_conv2d_and_relu(relu1_1, 2)
    pool4 = _build_pooling(relu1_2)
    conv2_1, relu2_1 = _build_conv2d_and_relu(pool4, 5)
    conv2_2, relu2_2 = _build_conv2d_and_relu(relu2_1, 7)
    pool9 = _build_pooling(relu2_2)
    conv3_1, relu3_1 = _build_conv2d_and_relu(pool9, 10)
    conv3_2, relu3_2 = _build_conv2d_and_relu(relu3_1, 12)
    conv3_3, relu3_3 = _build_conv2d_and_relu(relu3_2, 14)
    conv3_4, relu3_4 = _build_conv2d_and_relu(relu3_3, 16)
    pool18 = _build_pooling(relu3_4)
    conv4_1, relu4_1 = _build_conv2d_and_relu(pool18, 19)
    conv4_2, relu4_2 = _build_conv2d_and_relu(relu4_1, 21)
    conv4_3, relu4_3 = _build_conv2d_and_relu(relu4_2, 23)
    conv4_4, relu4_4 = _build_conv2d_and_relu(relu4_3, 25)
    pool27 = _build_pooling(relu4_4)
    conv5_1, relu5_1 = _build_conv2d_and_relu(pool27, 28)
    conv5_2, relu5_2 = _build_conv2d_and_relu(relu5_1, 30)
    conv5_3, relu5_3 = _build_conv2d_and_relu(relu5_2, 32)
    conv5_4, relu5_4 = _build_conv2d_and_relu(relu5_3, 34)
    pool36 = _build_pooling(relu5_4)  # this layer is not used in neural style transfer
    return network_input, conv4_2, relu1_1, relu2_1, relu3_1, relu4_1, relu5_1


def get_content_loss(c: tf.Tensor, g: tf.Tensor):
    _, h, w, ch = c.get_shape().as_list()
    c_reshape = tf.reshape(c, [1, ch * h * w])
    g_reshape = tf.reshape(g, [1, ch * h * w])
    # the content loss defined in the paper
    loss = 1 / 2 * tf.reduce_sum(tf.square(tf.subtract(c_reshape, g_reshape)))

    return loss


def gram_matrix(a: tf.Tensor):
    return tf.matmul(a, a, transpose_a=True)


def get_one_layer_style_loss(ta, tb):
    _, h, w, ch = ta.get_shape().as_list()
    a = tf.reshape(ta, [h * w, ch])
    b = tf.reshape(tb, [h * w, ch])
    ga = gram_matrix(a)
    gb = gram_matrix(b)
    return 1 / (4 * h * h * w * w * ch * ch) * tf.reduce_sum(tf.square(tf.subtract(ga, gb)))


def get_style_loss(s1_1, s2_1, s3_1, s4_1, s5_1, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1):
    # the style loss defined in the paper(including the one layer style loss)
    return 0.2 * get_one_layer_style_loss(conv1_1, s1_1) + \
           0.2 * get_one_layer_style_loss(conv2_1, s2_1) + \
           0.2 * get_one_layer_style_loss(conv3_1, s3_1) + \
           0.2 * get_one_layer_style_loss(conv4_1, s4_1) + \
           0.2 * get_one_layer_style_loss(conv5_1, s5_1)


def get_total_loss(content_loss, style_loss, alpha=2e-4, beta=1):
    return alpha * content_loss + beta * style_loss


def swap(H_shape, conv_K_argmax, unnormalized_style_patch_filters):
    phi_cs = np.zeros(H_shape)
    num_of_pathcs, h_p, w_p, _ = unnormalized_style_patch_filters.shape
    phi_cs_overlap_times = np.zeros(H_shape)
    conv_h, conv_w = conv_K_argmax.shape
    for h_now in range(conv_h):
        for w_now in range(conv_w):
            patch_now_idx = conv_K_argmax[h_now][w_now]
            patch_now = unnormalized_style_patch_filters[patch_now_idx]
            phi_cs[h_now:h_now + h_p, w_now:w_now + w_p, :] += patch_now
            phi_cs_overlap_times[h_now:h_now + h_p, w_now:w_now + w_p, :] += 1

    return phi_cs / phi_cs_overlap_times
