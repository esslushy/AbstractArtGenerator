#This file contains functions and other utilities for the neural network

import numpy as np
import tensorflow as tf
from tensorflow import layers
import cv2
from os import listdir
from os.path import join, isfile

#use matplotlib to read and process images
def getImages(directory):
    images = []
    tfImages = []
    allImages = [image for image in listdir(directory) if isfile(join(directory, image))]#gets all images from images folder
    for imageDir in allImages:
        if (imageDir[:2] == "._"):#sometimes the listdir will pick up meta data thats not needed
            continue
        img = cv2.imread(directory + imageDir)
        img = ((img - 127.5) / 127.5) #normalizes and converts images to -1 -> 1 range
        images.append(img)
        tfImages.append(tf.Variable(img))
    np.save("ImagesX64.npy", images)
    return tf.data.Dataset.from_tensor_slices((tfImages))

#load images from ready .npy file and cast to dataset
def loadImages(file):
    arr = np.load(file)
    arr = [tf.Variable(x) for x in arr]
    return tf.data.Dataset.from_tensor_slices((arr))

#reshapes the noise into a specific size
def linear(inputs, output_size, scope=None, mean=0., stddev=0.02, bias_start=0.0, with_w=False):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(mean=mean, stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            # import ipdb; ipdb.set_trace()
            return tf.matmul(inputs, matrix) + bias, matrix, bias
        else:
            return tf.matmul(inputs, matrix) + bias

def convTranspose(inputs, output, activate, generation):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001, scope="generator_regularizer")
    inputs = layers.conv2d_transpose(inputs=inputs, filters=output, kernel_size=5, 
                                    strides=(2,2), padding="same", 
                                    kernel_initializer=tf.initializers.glorot_uniform,
                                    name="g_deconvolutional_layer_" + str(generation), 
                                    bias_initializer=tf.constant_initializer(0.0),  
                                    bias_regularizer=regularizer,
                                    kernel_regularizer=regularizer)#use same for consistency. Bias is already on and starts at 0
    if activate:
        inputs = layers.batch_normalization(inputs=inputs, momentum=0.5, name="g_batch_normalizer_" + str(generation))# uses g for generator
        inputs = tf.nn.relu(inputs)
    return inputs

def conv(inputs, output, normalize, generation):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001, scope="discriminator_regularizer")
    inputs = layers.conv2d(inputs=inputs, filters=output, kernel_size=5, strides=(2, 2), padding="same",
                            kernel_initializer=tf.initializers.glorot_uniform, 
                            name="d_convolutional_layer_" + str(generation),
                            bias_initializer=tf.constant_initializer(0.0), 
                            bias_regularizer=regularizer,
                            kernel_regularizer=regularizer)
    if normalize:
        inputs = layers.batch_normalization(inputs=inputs, momentum=0.5, name="d_batch_normalizer_" + str(generation))#uses d for discriminator
    inputs = tf.nn.leaky_relu(inputs)
    return inputs
    
