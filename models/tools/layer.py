import numpy as np
import tensorflow as tf

from models.tools.activation_function import get_activation_fn
from models.tools.normalization_function import get_normalization_fn


def grayscale_to_rainbow(image):
    with tf.variable_scope('garyscale_to_rainbow'):
        H = tf.squeeze((1. - image) * 2. / 3., axis=-1)
        SV = tf.ones(H.get_shape())
        HSV = tf.stack([H, SV, SV], axis=3)
        RGB = tf.image.hsv_to_rgb(HSV)
    return RGB


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_rank(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.get_shape()) - 2


# Weight initialization (Xavier's init)
def xavier_initializer_convolution(shape, activation_type='sigmoid', is_uniform=True, lambda_initializer=False):
    with tf.device('/cpu:0'):
        s = len(shape) - 2
        # num_activations = shape[0] * shape[1] * shape[2] * shape[3] + shape[-1]
        num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
        if activation_type == 'sigmoid':
            if is_uniform:
                lim = np.sqrt(6. / num_activations)
                if lambda_initializer:
                    return np.random.uniform(-lim, lim, shape).astype(np.float32)
                else:
                    return tf.random_uniform(shape, minval=-lim, maxval=lim)
            else:
                stddev = np.sqrt(3. / num_activations)
                if lambda_initializer:
                    return np.random.normal(0, stddev, shape).astype(np.float32)
                else:
                    return tf.truncated_normal(shape, mean=0, stddev=stddev)
        elif activation_type in ['relu', 'lrelu', 'prelu']:
            if is_uniform:
                lim = np.sqrt(6. / num_activations) * np.sqrt(2)
                if lambda_initializer:
                    return np.random.uniform(-lim, lim, shape).astype(np.float32)
                else:
                    return tf.random_uniform(shape, minval=-lim, maxval=lim)
            else:
                stddev = np.sqrt(3. / num_activations) * np.sqrt(2)
                if lambda_initializer:
                    return np.random.normal(0, stddev, shape).astype(np.float32)
                else:
                    return tf.truncated_normal(shape, mean=0, stddev=stddev)
        elif activation_type == 'tan':
            if is_uniform:
                lim = np.sqrt(6. / num_activations) * 4
                if lambda_initializer:
                    return np.random.uniform(-lim, lim, shape).astype(np.float32)
                else:
                    return tf.random_uniform(shape, minval=-lim, maxval=lim)
            else:
                stddev = np.sqrt(3. / num_activations) * 4
                if lambda_initializer:
                    return np.random.normal(0, stddev, shape).astype(np.float32)
                else:
                    return tf.truncated_normal(shape, mean=0, stddev=stddev)
        raise ValueError('Distribution must be either "uniform" or "normal".')


def constant_initializer(value, shape, lambda_initializer=False):
    with tf.device('/cpu:0'):
        if lambda_initializer:
            return np.full(shape, value).astype(np.float32)
        else:
            return tf.constant(value, tf.float32, [shape])


def convolution(x, filter_, padding='SAME', strides=None, activation_type='relu'):
    w = tf.get_variable(name='weights',
                        initializer=xavier_initializer_convolution(shape=filter_, activation_type=activation_type))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0.1, shape=filter_[-1]))
    return tf.nn.convolution(x, w, padding, strides) + b


def deconvolution(x, filter_, strides, padding='SAME', activation_type='relu'):
    w = tf.get_variable(name='weights',
                        initializer=xavier_initializer_convolution(shape=filter_, activation_type=activation_type))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0.1, shape=filter_[-2]))
    spatial_rank = get_spatial_rank(x)
    shape = tf.shape(x)
    shape_list = x.get_shape().as_list()
    size = [shape[0] * strides[0]] + [shape_list[i] * strides[i] for i in range(1, len(shape_list) - 1)] + [
        shape[-1] // 2]
    output_shape = tf.stack(size)
    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


def crop_and_concat(x1, x2):
    # offsets for the top left corner of the crop
    _, *size1, _ = x1.get_shape().as_list()
    _, *size2, _ = x2.get_shape().as_list()
    if size1 != size2:
        offsets = [0] + [(size2[i] - size1[i]) // 2 for i in range(len(size1))] + [0]
        size = [-1, *size1, -1]
        x2_crop = tf.slice(x2, offsets, size)
        return tf.concat([x1, x2_crop], axis=-1)
    else:
        return tf.concat([x1, x2], axis=-1)


def resnet_add(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    if x1_shape[-1] != x2_shape[-1]:
        pad = [(0, 0) for _ in range(len(x1_shape) - 1)] + [(0, x1_shape[-1] - x2_shape[-1])]
        residual_connection = x1 + tf.pad(x2, pad)
    else:
        residual_connection = x1 + x2
    return residual_connection


def conv_bn_relu_drop(input_, filter_, strides=None, activation_type=None, norm_type=None, is_train=True, keep_prob=1,
                      replace=1, resnet=True):
    with tf.variable_scope('conv_bn_relu_drop'):
        x = input_
        for i in range(replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                x = convolution(x, filter_, strides=strides, activation_type=activation_type)
                x = get_normalization_fn(x, norm_type, is_train)
                x = get_activation_fn(x, activation_type)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
                if resnet and i == replace - 1:
                    x = resnet_add(x, input_)
        return x


def down_conv_bn_relu(x, filter_, strides, activation_type=None, norm_type=None, is_train=True):
    with tf.variable_scope('down_conv_bn_relu'):
        x = convolution(x, filter_, strides=strides)
        x = get_normalization_fn(x, norm_type, is_train)
        x = get_activation_fn(x, activation_type)
        return x


def deconv_bn_relu(x, filter_, strides, activation_type=None, norm_type=None, is_train=True):
    with tf.variable_scope('deconv_bn_relu'):
        x = deconvolution(x, filter_, strides=strides)
        x = get_normalization_fn(x, norm_type, is_train)
        x = get_activation_fn(x, activation_type)
        return x


def concat_conv_bn_relu_drop(input_, feature, filter_, strides, activation_type=None, norm_type=None, is_train=True,
                             keep_prob=1, replace=1, resnet=True):
    with tf.variable_scope('concat_conv_bn_relu_drop'):
        x = crop_and_concat(input_, feature)
        x = get_normalization_fn(x, norm_type, is_train)
        for i in range(replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                if i == 0:
                    _filter = [_f if _i != len(filter_) - 2 else _f * 2 for _i, _f in enumerate(filter_)]
                    x = convolution(x, _filter, strides=strides)
                    input_ = get_normalization_fn(x, norm_type, is_train, scope='input')
                else:
                    x = convolution(x, filter_, strides=strides)
                x = get_normalization_fn(x, norm_type, is_train)
                x = get_activation_fn(x, activation_type)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
                if resnet and i == replace - 1:
                    x = resnet_add(x, input_)
        return x
