#tensor needs this stuff
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import layers
import matplotlib.pyplot as plt
from ops import *

tf.logging.set_verbosity(tf.logging.INFO)#make tesnorflow log stuffs

#standardized variables
batch_size = 100 #each batch will have 100 images


#generator function. Takes a noise value of size (?, 100) and returns a tensor of 64x64x3
def generator(x):
    x = tf.reshape(linear(x, 1024 * 4 * 4), [x.shape[0], 4, 4, 1024])#in default NHWC batch height width inchannels
    #filter is the output shape where it doubles both width and height and halves filters
    x = convTranspose(x, 512, True)#1024 -> 512
    x = convTranspose(x, 256, True)#512 -> 256
    x = convTranspose(x, 128, True)#256 -> 128
    #don't activate for last layer of generator
    x = convTranspose(x, 3, False)#128 -> 3 because rgb coloration is 3 
    #finish the output with a tanh activation
    x = layers.dense(x, 3, activation=tf.nn.tanh)
    return x

#takes a parameter size which is how many 100 number long noise arrays it makes
def noise(size):
    return tf.Variable(tf.random_normal([size, 100]), expected_shape=[size, 100])


#displays images live
# for i in range(100):
#     img = np.random.random((50,50,3))
#     plt.imshow(img)
#     plt.pause(0.05)

x = noise(4)
print(generator(noise(4)))