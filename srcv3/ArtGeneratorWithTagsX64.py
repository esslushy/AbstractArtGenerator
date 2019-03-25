#imports
#load data
from torch.utils.data import dataloader, TensorDataset
import torch
#network
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, BatchNormalization
from tensorflow import nn

from ops import noise, deconvolutLayer, convolutLayer, convolutionalConcat

batchSize = 100
noiseLength = 100
tagLength = 20
numEpochs = 150
#standardize randomness
tf.set_random_seed(7)
#global step
globalStep = 1
#dataset
images = torch.from_numpy(np.load('C:\\Users\\evans\\datasets\\images.npy'))
tags = torch.from_numpy(np.load('C:\\Users\\evans\\datasets\\tags.npy'))
dataset = TensorDataset(images, tags)
dataLoader = dataloader.DataLoader(dataset, batch_size=batchSize, shuffle=True)
#models
def generator(z, y):
    yb = tf.reshape(y, [-1, 1, 1, tagLength])#makes 3d config of tags to append to conv
    with tf.variable_scope('generator'):
        with tf.variable_scope('reshape_and_flatten'):
            z = tf.concat(values=[z, y], axis=1)#connect along channels not batch
            z = Dense(units=1024*4*4, kernel_initializer=tf.contrib.layers.xavier_initializer())(z)
            z = tf.reshape(z, (-1, 4, 4, 1024))
        with tf.variable_scope('deconvolution_layer_1'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 512)#4x4x1024 -> 8x8x512
            z = BatchNormalization(axis=3)(z)#3 is channels axis
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_2'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 256)#8x8x512 -> 16x16x256
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_3'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 128)#16x16x256 -> 32x32x128
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_4'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 64)#32x32x128 -> 64x64x64
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_5'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 32, stride=1)#64x64x64 -> 64x64x32
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_6'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 16, stride=1)#64x64x32 -> 64x64x16
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_7'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 8, stride=1)#64x64x16 -> 64x64x8
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_8'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 4, stride=1)#64x64x8 -> 64x64x4
            z = BatchNormalization(axis=3)(z)
            z = nn.relu(z)
        with tf.variable_scope('deconvolution_layer_9'):
            z = convolutionalConcat(z, yb)
            z = deconvolutLayer(z, 3, stride=1)#64x64x4 -> 64x64x3
            print(z)
            #no relu or batch norm for output
        with tf.variable_scope('output'):
            out = nn.tanh(z)
            #output to show it off
            tf.summary.image('Generated Images', out, max_outputs=48)
    return out

def discriminator(x, y):
    yb = tf.reshape(y, [-1, 1, 1, tagLength])#makes 3d config of tags to append to conv
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('convolutional_layer_1'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 4, stride=1)#64x64x3 -> 64x64x4
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_2'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 8, stride=1)#64x64x4 -> 64x64x8
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_3'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 16, stride=1)#64x64x8 -> 64x64x16
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_4'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 32, stride=1)#64x64x16 -> 64x64x32
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_5'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 64, stride=1)#64x64x32 -> 64x64x64
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_6'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 128)#64x64x64 -> 32x32x128
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_7'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 256)#32x32x128 -> 16x16x256
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_8'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 512)#16x16x256 -> 8x8x512
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('convolutional_layer_9'):
            x = convolutionalConcat(x, yb)
            x = convolutLayer(x, 1024)#8x8x512 -> 4x4x1024
            x = BatchNormalization(axis=3)(x)
            x = nn.leaky_relu(x, alpha=0.2)
        with tf.variable_scope('flatten'):
            x = tf.reshape(x, (-1, 4*4*1024))#from 3d object to 1024x4x4 line
            x = tf.concat(values=[x, y], axis=1)
        with tf.variable_scope('output'):
            logits = Dense(units=1, kernel_initializer=tf.contrib.layers.xavier_initializer())(x)
            out = nn.sigmoid(x)
    return out, logits

#placeholders
x = tf.placeholder(dtype=tf.float16, shape=(None, 64, 64, 3), name='Images')
z = tf.placeholder(dtype=tf.float16, shape=(None, noiseLength), name='Noise')
y = tf.placeholder(dtype=tf.float16, shape=(None, tagLength), name='Tags')

device = ''
if tf.test.is_gpu_available():
    device = '/gpu:0'
else:
    device = '/cpu:0'
with tf.device(device):
    generatorSamples = generator(z, y)#make generator with z (noise) placeholder
    discriminatorReal, discriminatorRealLogits = discriminator(x, y)#make discriminator that takes data from the real
    discriminatorFake, discriminatorFakeLogits = discriminator(generatorSamples, y)#make discriminator that takes fake data

discriminatorLoss = tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorRealLogits, labels=tf.ones_like(discriminatorRealLogits),
                               name="discriminator_loss_real"
                               #takes real input and makes the labels 1 or real because it wants to identify real data as real
                           )
                    ) + tf.reduce_mean(
                           nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFakeLogits, labels=tf.zeros_like(discriminatorFakeLogits),
                               name="discriminator_loss_fake"
                               #takes fake input and makes the labels 0 or fake because it wants to identify fake data as fake
                           )
                        )
                    
generatorLoss = tf.reduce_mean(
                            nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminatorFakeLogits, labels=tf.ones_like(discriminatorFakeLogits) * .9,
                               name="generator_loss_real"
                               #takes fake input and makes the labels 0 or fake because it wants to identify fake data as fake
                            )
                        )
#write losses to tensorboard
tf.summary.scalar("Discriminator Total Loss", discriminatorLoss)
tf.summary.scalar("Generator Loss", generatorLoss)

#Optimzer setup
trainableVariables = tf.trainable_variables()
#seperate trainable variables into ones for discriminator and generator
dTrainableVariables = [var for var in trainableVariables if "discriminator" in var.name]
gTrainableVariables = [var for var in trainableVariables if "generator" in var.name]

#build adam optimizers. paper said to use .0002. discriminator a tad strong so used .0001. 256 version used smaller numbers b/c smaller batch
learningRate = tf.train.exponential_decay(.001, globalStep,
                                           1000, 0.96, staircase=True)#decays learning rate b .96 every 100k steps
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    discriminatorOptimizer = tf.train.AdamOptimizer(learning_rate=learningRate, beta1=0.5).minimize(discriminatorLoss, var_list=gTrainableVariables)#epsilon is already the same
    generatorOptimizer = tf.train.AdamOptimizer(learning_rate=.002, beta1=0.5).minimize(generatorLoss, var_list=gTrainableVariables)#epsilon is already the same


#config
config = tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3, allow_soft_placement=True)

#Saver for when stuff goes wrong
saver = tf.train.Saver()

#merge summaries
merged = tf.summary.merge_all()

#session
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("./info", sess.graph)
    sess.run(tf.global_variables_initializer())
    print("Starting Session")
    for epoch in range(numEpochs):
        for numBatch, (images, tags) in enumerate(dataLoader):

            images = images[0].numpy() #turns them into numpy and sticks them into another array
            tags = tags[0].numpy()
            summary, _, _ = sess.run([merged, generatorOptimizer, discriminatorOptimizer], feed_dict={ x : images, z : noise(batchSize, noiseLength),  y : tags })
            if numBatch % 10 == 0:
                writer.add_summary(summary, globalStep)
                globalStep+=1
            if numBatch % 100 == 0:
                saver.save(sess, "./model/DCGAN_Epoch_%s_Batch_%s.ckpt" % (epoch, numBatch))
    saver.save(sess, './model/FullyTrainedModel')          
    writer.close()