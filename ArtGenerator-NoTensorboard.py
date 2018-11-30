#These imports are used for the dataset because torch has a better implementation of getting the data
from torch.utils.data import dataloader
from torchvision import transforms, datasets

#These imports are for the network itself
import tensorflow as tf
from tensorflow import nn, layers
from tensorflow.contrib import layers as contribLayers
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

"""Load and Prepare Data"""
dataset = getCifarDataset()
batchSize = 100
dataLoader = dataloader.DataLoader(dataset, batch_size=batchSize, shuffle=True)
numBatches = len(dataLoader)
imageShape = (64, 64, 3)
noiseLength = 100
numEpochs = 200
#normalize randomness
tf.set_random_seed(7)

"""Models"""
#regularizers (keeps trainign stable)
generatorRegularizer = contribLayers.l2_regularizer(0.0001, scope="generator_regularizer")
discriminatorRegularizer = contribLayers.l2_regularizer(0.0001, scope="discriminator_regularizer")

#takes image x and ouputs a value between 0 and 1 where 0 is fake and 1 is real
def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("convolutional_layer_1"):
            x = convolutLayer(x, 128, discriminatorRegularizer)#3x64x64 -> 128x32x32
            #don't batch normalize in first layer of discriminator
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_2"):
            x = convolutLayer(x, 256, discriminatorRegularizer)#128x32x32 -> 256x16x16
            x = layers.batch_normalization(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_3"):
            x = convolutLayer(x, 512, discriminatorRegularizer)#256x16x16 -> 512x8x8
            x = layers.batch_normalization(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolution_layer_4"):
            x = convolutLayer(x, 1024, discriminatorRegularizer)#512x8x8 -> 1024x4x4
            x = layers.batch_normalization(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("linear"):
            x = contribLayers.flatten(x)#from 3d object 1024x4x4
            x = contribLayers.fully_connected(x, 1)#1024x4x4 to a shape of 1 to sigmoid
        with tf.variable_scope("output"):
            out = nn.sigmoid(x)
    return out

def generator(z):
    with tf.variable_scope("generator"):
        with tf.variable_scope("reshape_and_flatten"):
            z = contribLayers.fully_connected(z, 1024 * 4 * 4)#flatten
            z = tf.reshape(z, (-1, 4, 4, 1024))#reshape
        with tf.variable_scope("deconvolutional_layer_1"):
            z = deconvolutLayer(z, 512, discriminatorRegularizer)#1024x4x4 -> 512x8x8
            z = layers.batch_normalization(z)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_2"):
            z = deconvolutLayer(z, 256, discriminatorRegularizer)#512x8x8 -> 256x16x16
            z = layers.batch_normalization(z)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_3"):
            z = deconvolutLayer(z, 128, discriminatorRegularizer)#256x16x16 -> 128x32x32
            z = layers.batch_normalization(z)
            z = nn.relu(z)
        with tf.variable_scope("deconvolutional_layer_4"):
            z = deconvolutLayer(z, 3, discriminatorRegularizer)#128x32x32 -> 3x64x64
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
if tf.test.is_gpu_available(cuda_only=True):
    device = "/gpu:0"
else:
    device = "/cpu:0"
with tf.device(device):
    generatorSamples = generator(z)#make generator with z (noise) placeholder
    discriminatorReal = discriminator(x)#make discriminator that takes data from the real
    discriminatorFake = discriminator(generatorSamples)#make discriminator that takes fake data

discriminatorLossReal = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorReal, labels=tf.ones_like(discriminatorReal),
                               name="discriminator_loss_real"
                               #takes real input and makes the labels 1 or real because it wants to identify real data as real
                           ) 
                        )

discriminatorLossFake = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFake, labels=tf.zeros_like(discriminatorFake),
                               name="discriminator_loss_fake"
                               #takes fake input and makes the labels 0 or fake because it wants to identify fake data as fake
                           ) 
                        )

discriminatorRegularizerLoss = getRegularizerLoss("discriminator")

generatorLoss = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFake, labels=tf.ones_like(discriminatorFake),
                               name="generator_loss"
                               #takes fake input and makes the labels 1 or real because generator wants to make its fake dat seem more real
                           ) 
                        )

generatorRegularizerLoss = getRegularizerLoss("generator")

discriminatorTotalLoss = discriminatorLossReal + discriminatorLossFake + discriminatorRegularizerLoss
generatorTotalLoss = generatorLoss + generatorRegularizerLoss

#Optimzers
trainableVariables = tf.trainable_variables()
#seperate trainable variables into ones for discriminator and generator
dTrainableVariables = [var for var in trainableVariables if "discriminator" in var.name]
gTrainableVariables = [var for var in trainableVariables if "generator" in var.name]

#build adam optimizers. paper said to use .0002
#might add ones for real and fake loss
discriminatorOptimizer = tf.train.AdamOptimizer(0.0002).minimize(discriminatorTotalLoss, var_list=dTrainableVariables)
generatorOptimizer = tf.train.AdamOptimizer(0.0002).minimize(generatorTotalLoss, var_list=gTrainableVariables)

#Test Data. Used to show generator results
testNoise = noise(16)

#config for session with multithreading, but limit to 3 of my 4 CPUs (tensor uses all by default: https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device)
config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1, log_device_placement=True)

#Saver for when stuff goes wrong
saver = tf.train.Saver()

#merge summaries
merged = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    i = 1
    for epoch in range(numEpochs):
        for numBatch, (realData, _) in enumerate(dataLoader):
            
            #Prepare the data. Since torch does it by Channels Height Width, but tensor takes Height Width Channels
            realData = realData.permute(0, 2, 3, 1).numpy()#convert to numpy array

            #Train Discriminator. The values are none, the loss, an array of all real prediction, an array of all fake predictions
            _, dLoss, dRealPred, dFakePred, = sess.run([discriminatorOptimizer, discriminatorTotalLoss, discriminatorReal, discriminatorFake], 
                                    feed_dict={ x : realData, z : noise(batchSize) })
            
            #Train Generator. The values are none and loss
            _, gLoss = sess.run([generatorOptimizer, generatorTotalLoss],
                                feed_dict={ z: noise(batchSize) })

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
                