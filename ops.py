#This file contains functions and other utilities for the neural network

import tensorflow as tf
from tensorflow import layers

#reshapes the noise into a specific size
def linear(input_, output_size, scope=None, mean=0., stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(mean=mean, stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            # import ipdb; ipdb.set_trace()
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def convTranspose(inputs, output, activate):
    inputs = layers.conv2d_transpose(inputs=inputs, filters=output, kernel_size=4, strides=(2,2), use_bias=False, padding="same")#use same for consistency
    inputs = layers.batch_normalization(inputs=inputs)
    if activate:
        inputs = layers.dense(inputs=inputs, units=output, activation=tf.nn.relu)
    return inputs
    
