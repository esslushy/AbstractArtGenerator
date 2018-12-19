from ops import *
import tensorflow as tf
from tensorflow import nn, layers
import matplotlib.pyplot as plt

"""Models"""
#takes image x and ouputs a value between 0 and 1 where 0 is fake and 1 is real
def discriminator(x):#might be too powerful, already lowered learning rate, but might need to add dropout also
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("convolutional_layer_1"):
            x = convolutLayer(x, 128)#3x64x64 -> 128x32x32
            #don't batch normalize in first layer of discriminator
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_2"):
            x = convolutLayer(x, 256)#128x32x32 -> 256x16x16
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_3"):
            x = convolutLayer(x, 512)#256x16x16 -> 512x8x8
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolution_layer_4"):
            x = convolutLayer(x, 1024)#512x8x8 -> 1024x4x4
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("flatten"):
            x = tf.reshape(x, (-1, 4, 4, 1024))#from 3d object 1024x4x4
            logits = layers.dense(x, units=1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#1024x4x4 to a shape of 1 to sigmoid
        with tf.variable_scope("output"):
            out = nn.sigmoid(x)
    return out, logits #needed for loss function otherwise you double sigmoid

def generator(z):
    with tf.variable_scope("generator"):
        with tf.variable_scope("reshape_and_flatten"):
            z = layers.dense(inputs=z, units=4*4*1024)#flatten
            z = tf.reshape(z, (-1, 4, 4, 1024))#reshape noise 
        with tf.variable_scope("deconvolutional_layer_1"):
            z = deconvolutLayer(z, 512)#1024x4x4 -> 512x8x8
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_2"):
            z = deconvolutLayer(z, 256)#512x8x8 -> 256x16x16
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_3"):
            z = deconvolutLayer(z, 128)#256x16x16 -> 128x32x32
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_4"):
            z = deconvolutLayer(z, 3)#128x32x32 -> 3x64x64
            #no batch normalization in last layer of generatror
            #don't use relu for output
        with tf.variable_scope("output"):
            out = nn.tanh(z)
            #output to show it off
            tf.summary.image("Generated Images", out, max_outputs=16)
    return out



#makes input into discriminator a 4d array where it is array of 3d arrays to represent images
x = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3), name="Images")
#noise unkown length for unkown number of images to be made
z = tf.placeholder(dtype=tf.float32, shape=(None, 100), name="Noise")

with tf.device("/cpu:0"):
    generatorSamples = generator(z)#make generator with z (noise) placeholder
    discriminatorReal, discriminatorRealLogits = discriminator(x)#make discriminator that takes data from the real
    discriminatorFake, discriminatorFakeLogits = discriminator(generatorSamples)#make discriminator that takes fake data

#reloader 
reloader = tf.train.Saver()

with tf.Session() as sess:
    reloader.restore(sess, "./model/DCGAN_Epoch_%s_Batch_%s.ckpt" % (3, 700))
    plt.imshow(denormalize(sess.run(generatorSamples, feed_dict={ z: noise(1) })[0]))
    plt.show()