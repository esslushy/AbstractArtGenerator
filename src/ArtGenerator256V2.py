#These imports are used for the dataset because torch has a better implementation of getting the data
from torch.utils.data import dataloader, TensorDataset
from torchvision import transforms, datasets
import torch

#These imports are for the network itself
import tensorflow as tf
from tensorflow import nn, layers
import numpy as np
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
    return TensorDataset(torch.from_numpy(loadMemMap(file)))

"""Load and Prepare Data"""
batchSize = 32
noiseLength = 100
numEpochs = 150
unrollingSteps = 10
#standardize randomness
tf.set_random_seed(7)
#set global step
globalStep = 1

#make dataset loader since dataset split into multiple parts, you need to load different versions
def loadDataset(file):
    dataset = getCustomDataset(file)#gets data
    dataLoader = dataloader.DataLoader(dataset, batch_size=batchSize, shuffle=True)#splits it into batches
    return dataLoader

"""Models"""
#takes image x and ouputs a value between 0 and 1 where 0 is fake and 1 is real
def discriminator(x):#might be too powerful, already lowered learning rate, but might need to add dropout also
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("convolutional_layer_1"):
            x = convolutLayer(x, 4, (1,1))#3x256x256 -> 4x256x256
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_2"):
            x = convolutLayer(x, 8, (1,1))#4x256x256 -> 8x256x256
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_3"):
            x = convolutLayer(x, 16, (1,1))#8x256x256 -> 16x256x256
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_4"):
            x = convolutLayer(x, 32)#16x256x256 -> 32x128x128
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_5"):
            x = convolutLayer(x, 64)#32x128x128 -> 64x64x64
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_6"):
            x = convolutLayer(x, 128)#64x64x64 -> 128x32x32
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_7"):
            x = convolutLayer(x, 256)#128x32x32 -> 256x16x16
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolutional_layer_8"):
            x = convolutLayer(x, 512)#256x16x16 -> 512x8x8
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("convolution_layer_9"):
            x = convolutLayer(x, 1024)#512x8x8 -> 1024x4x4
            x = layers.batch_normalization(x, training=True)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope("flatten"):
            x = tf.reshape(x, (-1, 4*4*1024))#from 3d object 1024x4x4
            logits = layers.dense(x, units=1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#1024x4x4 to a shape of 1 to sigmoid
        with tf.variable_scope("output"):
            out = nn.sigmoid(x)
    return out, logits #needed for loss function otherwise you double sigmoid

def generator(z):
    with tf.variable_scope("generator"):
        with tf.variable_scope("reshape_and_flatten"):
            z = layers.dense(inputs=z, units=4*4*1024)#flatten
            z = tf.reshape(z, (-1, 4, 4, 1024))#reshape noise 
        with tf.variable_scope("deconvolution_layer_1"):
            z = deconvolutLayer(z, 512)#1024x4x4 -> 512x8x8
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_2"):
            z = deconvolutLayer(z, 256)#512x8x8 -> 256x16x16
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_3"):
            z = deconvolutLayer(z, 128)#256x16x16 -> 128x32x32
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_4"):
            z = deconvolutLayer(z, 64)#128x32x32 -> 64x64x64
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_5"):
            z = deconvolutLayer(z, 32)#64x64x64 -> 32x128x128
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_6"):
            z = deconvolutLayer(z, 16)#32x128x128 -> 16x256x256
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_7"):
            z = deconvolutLayer(z, 8, (1,1))#16x256x256 -> 8x256x256
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_8"):
            z = deconvolutLayer(z, 4, (1,1))#8x256x256 -> 4x256x256
            z = layers.batch_normalization(z, training=True)
            z = nn.relu(z)
        with tf.variable_scope("deconvolution_layer_9"):
            z = deconvolutLayer(z, 3, (1,1))#4x256x256 -> 3x256x256
            #no batch norm in last layer
            #relu not used for output
        with tf.variable_scope("output"):
            out = nn.tanh(z)
            #output to show it off
            tf.summary.image("Generated Images", out, max_outputs=16)
    return out

#Placeholders
#makes input into discriminator a 4d array where it is array of 3d arrays to represent images
x = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name="Images")
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

discriminatorLoss = -tf.reduce_mean(tf.math.log(nn.sigmoid(discriminatorRealLogits - discriminatorFakeLogits) + 1e-4) * .5)       
generatorLoss = -tf.reduce_mean(tf.math.log(nn.sigmoid(discriminatorFakeLogits - discriminatorRealLogits) + 1e-4) * .5)
#write losses to tensorboard
tf.summary.scalar("Discriminator Total Loss", discriminatorLoss)
tf.summary.scalar("Generator Loss", generatorLoss)

#Optimzer setup
trainableVariables = tf.trainable_variables()
#seperate trainable variables into ones for discriminator and generator
dTrainableVariables = [var for var in trainableVariables if "discriminator" in var.name]
gTrainableVariables = [var for var in trainableVariables if "generator" in var.name]

#build adam optimizers. paper said to use .0002. discriminator a tad strong so used .0001. 256 version used smaller numbers b/c smaller batch
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    discriminatorOptimizer = tf.train.AdamOptimizer(learning_rate=.002, beta1=0.5).minimize(discriminatorLoss, var_list=gTrainableVariables)#epsilon is already the same
    generatorOptimizer = tf.train.AdamOptimizer(learning_rate=.002, beta1=0.5).minimize(generatorLoss, var_list=gTrainableVariables)#epsilon is already the same

#config for session with multithreading, but limit to 3 of my 4 CPUs (tensor uses all by default: https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device)
config = tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3, allow_soft_placement=True)

#Saver for when stuff goes wrong
saver = tf.train.Saver()

#merge summaries
merged = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("./info256v2", sess.graph)
    sess.run(tf.global_variables_initializer())
    print("Starting Session")
    dataLoader = loadDataset("ImagesX256.npy")
    for epoch in range(numEpochs):
        for numBatch, realData in enumerate(dataLoader):

            realData = realData[0].numpy() #turns them into numpy and sticks them into another array
                
            summary, _, _ = sess.run([merged, generatorOptimizer, discriminatorOptimizer], feed_dict={ x : realData, z : noise(len(realData), noiseLength) })
            if numBatch % 10 == 0:
                writer.add_summary(summary, globalStep)
                globalStep+=1
            if numBatch % 100 == 0:
                saver.save(sess, "./model256v2/DCGAN_Epoch_%s_Batch_%s.ckpt" % (epoch, numBatch))
    saver.save(sess, "./model256v2/DCGAN_Final")          
    writer.close()