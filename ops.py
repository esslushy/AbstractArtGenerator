#imports
import tensorflow as tf
from tensorflow import layers
import numpy as np

#standard of numpy randomness
np.random.seed(7)

def convolutLayer(inputs, outputShape, regularizer):
    return layers.conv2d(inputs=inputs, filters=outputShape, kernel_size=4, strides=(2, 2), padding="same",
                            data_format="channels_last", use_bias=True, bias_initializer=tf.constant_initializer(0),
                            kernel_regularizer=regularizer, kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconvolutLayer(inputs, outputShape, regularizer):
    return layers.conv2d_transpose(inputs=inputs, filters=outputShape, kernel_size=4, strides=(2,2), 
                                    padding="same", data_format="channels_last", use_bias=True,
                                    bias_initializer=tf.constant_initializer(0), kernel_regularizer=regularizer,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

def noise(size):
    return np.random.normal(size=(size, 100))#noise is always 100 long

def denormalize(images):
    #change to 0 -> 1 ((x + currentMin) / (currentMax - currentMin)) * (newMax - newMin) + newMin
    images = (images+1) / 2
    return images

def getRegularizerLoss(regscope):
    return tf.add_n(tf.losses.get_regularization_losses(scope=regscope))