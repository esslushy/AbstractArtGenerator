#These imports are used for the dataset because torch has a better implementation of getting the data
from torch.utils.data import dataloader, TensorDataset
from torchvision import transforms, datasets
import torch

#These imports are for the network itself
import tensorflow as tf
from tensorflow import nn, layers
import numpy as np
#to show test images
import matplotlib.pyplot as plt
#personal files
from ops import *

def getCifarDataset():
    compose = transforms.Compose([
        transforms.Resize(64), 
        transforms.ToTensor(), #converts an image from 0 -> 255 to 0 -> 1 and into a floatTensor of channels x height x width
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))])#normalizes between -1 and 1 because that what generator outputs
        #works because subtracts each rgb channel by .5 and divides by .5 so least is -1 and most is 1
    return datasets.CIFAR10(root="./dataset", train=True, transform=compose, download=True)

def getCustomDataset(file):
    return TensorDataset(torch.from_numpy(loadImages(file)))

"""Load and Prepare Data"""
dataset = getCustomDataset("ImagesX64.npy")
batchSize = 256
dataLoader = dataloader.DataLoader(dataset, batch_size=batchSize, shuffle=True)
numBatches = len(dataLoader)
imageShape = (64, 64, 3)
noiseLength = 100
numEpochs = 200
#normalize randomness
tf.set_random_seed(7)

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

#Placeholders
#makes input into discriminator a 4d array where it is array of 3d arrays to represent images
x = tf.placeholder(dtype=tf.float32, shape=(None, ) + imageShape, name="Images")
#noise unkown length for unkown number of images to be made
z = tf.placeholder(dtype=tf.float32, shape=(None, noiseLength), name="Noise")

#Make models and set to use gpu
#test if gpu is available
device = ""
if tf.test.is_gpu_available():
    device = "/gpu:0"
else:
    device = "/cpu:0"
with tf.device(device):
    generatorSamples = generator(z)#make generator with z (noise) placeholder
    discriminatorReal, discriminatorRealLogits = discriminator(x)#make discriminator that takes data from the real
    discriminatorFake, discriminatorFakeLogits = discriminator(generatorSamples)#make discriminator that takes fake data

discriminatorLossReal = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorRealLogits, labels=tf.ones_like(discriminatorRealLogits),
                               name="discriminator_loss_real"
                               #takes real input and makes the labels 1 or real because it wants to identify real data as real
                           ) 
                        )

discriminatorLossFake = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFakeLogits, labels=tf.zeros_like(discriminatorFakeLogits),
                               name="discriminator_loss_fake"
                               #takes fake input and makes the labels 0 or fake because it wants to identify fake data as fake
                           )
                        )

generatorLoss = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFakeLogits, labels=tf.ones_like(discriminatorFakeLogits),
                               name="generator_loss"
                               #takes fake input and makes the labels 1 or real because generator wants to make its fake dat seem more real
                           ) 
                        )

discriminatorTotalLoss = discriminatorLossReal + discriminatorLossFake

#Optimzer setup
trainableVariables = tf.trainable_variables()
#seperate trainable variables into ones for discriminator and generator
dTrainableVariables = [var for var in trainableVariables if "discriminator" in var.name]
gTrainableVariables = [var for var in trainableVariables if "generator" in var.name]

#build adam optimizers. paper said to use .0002. discriminator a tad strong so used .0001
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    discriminatorOptimizer = tf.train.AdamOptimizer(0.00015).minimize(discriminatorTotalLoss, var_list=dTrainableVariables)
    generatorOptimizer = tf.train.AdamOptimizer(0.0002).minimize(generatorLoss, var_list=gTrainableVariables)

#config for session with multithreading, but limit to 3 of my 4 CPUs (tensor uses all by default: https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device)
config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1, allow_soft_placement=True, log_device_placement=True)

#Test noise
testNoise = noise(16)

#Saver for when stuff goes wrong
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        for numBatch, realData in enumerate(dataLoader):

            realData = realData[0].numpy()#convert to numpy array

            #Train Discriminator and Generator. The values are none, the loss, an array of all real prediction, an array of all fake predictions
            #none, and generator loss
            _, dLoss, dRealPred, dFakePred, _, gLoss = sess.run([discriminatorOptimizer, discriminatorTotalLoss, 
                                                                discriminatorReal, discriminatorFake, generatorOptimizer, 
                                                                generatorLoss], 
                                                                feed_dict={ x : realData, z : noise(batchSize) })
            #print losses
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, numEpochs, numBatch, numBatches))
            print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dLoss, gLoss))
            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(dRealPred.mean(), dFakePred.mean()))

            #Save model and data and log predictions
            if numBatch % 100 == 0:
                saver.save(sess, "./model/DCGAN_Epoch_%s_Batch_%s.ckpt" % (epoch, numBatch))
                #show most recent with model save move out of this if you want constant checking
                testImages = sess.run(generatorSamples, feed_dict={ z : testNoise })
                #images are height, width, rgb value
                testImages = denormalize(testImages)
                for i in range(1, len(testImages)+1):
                    plt.subplot(4, 4, i)
                    plt.imshow(testImages[i-1])
                plt.pause(1)
                
