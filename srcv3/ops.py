import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose

def convolutLayer(inputs, outputShape, kernel=4, stride=2, padding='same', activation='linear', bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                 bias_inititializer=tf.constant_initializer(0)):
    return Conv2D(filters=outputShape, kernel_size=kernel, strides=stride, padding=padding, activation=activation, use_bias=bias, 
                    kernel_initializer=kernel_initializer, bias_initializer=bias_inititializer)(inputs)

def deconvolutLayer(inputs, outputShape, kernel=4, stride=2, padding='same', activation='linear', bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                 bias_inititializer=tf.constant_initializer(0)):
    return Conv2DTranspose(filters=outputShape, kernel_size=kernel, strides=stride, padding=padding, activation=activation, use_bias=bias, 
                    kernel_initializer=kernel_initializer, bias_initializer=bias_inititializer)(inputs)

def convolutionalConcat(inputs, tags):
    inputShapes = inputs.get_shape()
    tagShapes = tags.get_shape()
    return tf.concat(values=[inputs, tf.reshape(tags,[-1, inputShapes[1], inputShapes[2], tagShapes[3]])], axis=3)

def noise(size, length):
    return np.random.normal(size=(size, length))
    
def denormalize(images):
    #change to 0 -> 1 ((x + currentMin) / (currentMax - currentMin)) * (newMax - newMin) + newMin
    images = (images+1) / 2
    return images
