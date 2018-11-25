#tensor needs this stuff
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import layers
import matplotlib.pyplot as plt
from ops import *

tf.logging.set_verbosity(tf.logging.INFO)#make tesnorflow log stuffs
tf.set_random_seed(7)#constant seed so rng won't affect from different runs

#standardized variables
batchSize = 100 #each batch will have 100 images
numEpochs = 200#number of training epochs
showBatchSize = 16#number of test images

#prepare data
dataset = loadImages("ImagesX64.npy")
batchedDataset = dataset.shuffle(93000).batch(batchSize)
iterator = batchedDataset.make_initializable_iterator()
nextBatch = iterator.get_next()


#generator function. Takes a noise value of size (?, 100) and returns a tensor of 64x64x3
def makeGenerator(x):
    with tf.variable_scope("generator") as scope:
        #project and reshape from 100 to 1024x4x4
        x = linear(x, 1024 * 4 * 4, "g_linear")
        x = tf.reshape(x, [-1, 4, 4, 1024])#in default NHWC batch height width inchannels
        #filter is the output shape where it doubles both width and height and halves filters
        x = convTranspose(x, 512, True, 0)#1024 -> 512
        x = convTranspose(x, 256, True, 1)#512 -> 256
        x = convTranspose(x, 128, True, 2)#256 -> 128
        #don't activate or normalize for last layer of generator
        x = convTranspose(x, 3, False, 3)#128 -> 3 because rgb coloration is 3 
        #finish the output with a tanh activation
        x = tf.tanh(x)
        return x

#takes a 3d array of an image and returns a value between 0 and 1  where 0 is fake and 1 is real
def makeDiscriminator(x):
    with tf.variable_scope("discriminator") as scope:
        #dont normalize on first layer
        x = conv(x, 128, False, 0)#3 -> 128
        x = conv(x, 256, True, 1)#128 -> 256
        x = conv(x, 512, True, 2)#256 -> 512
        x = conv(x, 1024, True, 3)#512 -> 1024
        x = tf.reshape(x, [-1, 1024 * 4* 4])
        x = linear(x, 1, "d_linear")
        return tf.sigmoid(x), x


#takes a parameter size which is how many 100 number long noise arrays it makes
def noise(size):
    return tf.Variable(tf.random_normal([size, 100]), expected_shape=[size, 100])

def getRegularizerLoss(scope):
    return tf.add_n(tf.losses.get_regularization_losses(scope=scope))

with tf.device("/gpu:0"):
    z = tf.placeholder(tf.float32, shape=[None, 100], name="z")#holds noise
    generator = makeGenerator(z)#make generator
    images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="images")
    discriminator, discriminatorLogits = makeDiscriminator(images)#make discriminator. logits aka output
    showNoise = noise(showBatchSize)#used to show what generator has made after batchs
#allows program to use multiple cores
config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
#sets up config if gpu is available
if(tf.test.is_gpu_available):
    config.gpu_options.allow_growth=True

def makeFakeImages(sess, size):
    fakeNoise = noise(size)
    sess.run(fakeNoise.initializer)
    return sess.run(generator, feed_dict={ z : fakeNoise.eval() })

def train(sess, realData, fakeData, optimizer, discriminator, trainableVariables, gOptimizer, generator, gTrainableVariables):
    """Discriminator"""
    #reset gradients
    reset_optimizer_op = tf.variables_initializer(optimizer.variables())
    sess.run(reset_optimizer_op)
    #Train on real data
    predictionReal, predictionRealLogits = sess.run(discriminator,  feed_dict={ images : realData })
    #Calculate Loss
    lossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionRealLogits, labels=tf.ones_like(predictionReal)))
    
    #Train on fake data
    predictionFake, predictionFakeLogits = sess.run(discriminator, feed_dict={ images : fakeData})
    #Calculate Loss
    lossFake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionFakeLogits, labels=tf.zeros_like(predictionFake)))
    
    lossRegularizer = getRegularizerLoss("discriminator")  
    
    totalLoss = lossReal + lossFake + lossRegularizer
    
    #Backpropagate and update weights
    optimizer.minimize(totalLoss, var_list=trainableVariables)
    
    """Generator""" #uses previous fake data
    #Reset gradients
    reset_optimizer_op = tf.variables_initializer(gOptimizer.variables())
    sess.run(reset_optimizer_op)
    #calculate loss
    gLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionFakeLogits, labels=tf.ones_like(predictionFake)))#use ones because generator wants to be real
    
    totalGLoss = gLoss + getRegularizerLoss("generator")
    #backpropagate and update weights
    gOptimizer.minimize(totalGLoss, var_list=gTrainableVariables)
  
    return totalLoss, totalGLoss

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    #get trainable variables
    trainableVariables = tf.trainable_variables()
    generatorTrainableVariables = [var for var in trainableVariables if "g_" in var.name]#gets the variables to train generator on
    discriminatorTrainableVariables = [var for var in trainableVariables if "d_" in var.name]#same, but discriminator
    #to save model when everything goes wrong
    saver = tf.train.Saver()
    saveDirectory = "./training"
    """Train models"""
    #Optimizers
    discriminatorOptimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    generatorOptimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)

    for epoch in range(1, numEpochs+1):
        sess.run(iterator.initializer)
        while True:
            try:
                realImages = sess.run(nextBatch)
                realImages = [cv2.resize(img, (64, 64)) for img in realImages]
                fakeImages = makeFakeImages(sess, len(realImages))
                #train discriminator and generator
                discriminatorLoss, generatorLoss = train(sess, realImages, fakeImages, discriminatorOptimizer, discriminator, discriminatorTrainableVariables, generatorOptimizer, generator, generatorTrainableVariables)
                print("Discriminator Loss:  " + str(discriminatorLoss.eval()) + "\tGenerator Loss:  " + str(generatorLoss.eval()))
            except tf.errors.OutOfRangeError:
                break
        #show images after each run
        testImages = sess.run(generator, feed_dict={ z : showNoise.eval()})
        fig = plt.figure()
        for i in range(1, showBatchSize+1):
            fig.add_subplot(4, 4, i)
            plt.imshow((testImages[i-1] * 127.5) + 127.5)
        plt.pause(1)