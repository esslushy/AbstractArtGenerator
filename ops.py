#imports
import tensorflow as tf
from tensorflow import layers
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile

#standard of numpy randomness
np.random.seed(7)

#use matplotlib to read and process images
def getImages(directory, name):
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
    np.save(name, images)
    return tf.data.Dataset.from_tensor_slices((tfImages))

#load images from ready .npy file and cast to dataset
def loadImages(file):
    arr = np.load(file)
    return arr

def convolutLayer(inputs, outputShape):
    return layers.conv2d(inputs=inputs, filters=outputShape, kernel_size=4, strides=(2, 2), padding="same",
                            data_format="channels_last", use_bias=True, bias_initializer=tf.constant_initializer(0),
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconvolutLayer(inputs, outputShape):
    return layers.conv2d_transpose(inputs=inputs, filters=outputShape, kernel_size=4, strides=(2,2), 
                                    padding="same", data_format="channels_last", use_bias=True,
                                    bias_initializer=tf.constant_initializer(0),
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

def noise(size):
    return np.random.normal(size=(size, 100))#noise is always 100 long

def denormalize(images):
    #change to 0 -> 1 ((x + currentMin) / (currentMax - currentMin)) * (newMax - newMin) + newMin
    images = (images+1) / 2
    return images
