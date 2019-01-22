#imports
import tensorflow as tf
from tensorflow import layers
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile
from PIL import Image

#standard of numpy randomness
np.random.seed(7)

#use matplotlib to read and process images
def getImages(directory, name):
    images = []
    i=0
    allImages = [image for image in listdir(directory) if isfile(join(directory, image))]#gets all images from images folder
    for num, imageDir in enumerate(allImages):
        if (imageDir[:2] == "._"):#sometimes the listdir will pick up meta data thats not needed
            continue
        img = Image.open(join(directory, imageDir))
        img = np.array(img)
        img = img.astype('float32')
        #normalizes and converts images to -1 -> 1 range
        np.subtract(img, np.array(127.5), out=img)
        np.divide(img, np.array(127.5), out=img)
        images.append(img)
        if((num%20000)==0 and num != 0):
            np.save(name+str(i), images)
            images = []
            i+=1
    np.save(name+str(i), images)#save final set of images

#load images from ready .npy file and cast to dataset
def loadImages(file):
    arr = np.load(file)
    return arr

def loadMemMap(file):
    return np.memmap(file, dtype='float32', mode='r', shape=(93973, 256, 256, 3))

def resizeConvolutLayer(inputs, outputShape, size):
    inputs = tf.image.resize_image_with_pad(inputs, size, size)
    return layers.conv2d(inputs=inputs, filters=outputShape, kernel_size=4, strides=(1, 1), padding="same",
                            data_format="channels_last", use_bias=True, bias_initializer=tf.constant_initializer(0),
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

def convolutLayer(inputs, outputShape, stride=(2,2)):
    return layers.conv2d(inputs=inputs, filters=outputShape, kernel_size=4, strides=stride, padding="same",
                            data_format="channels_last", use_bias=True, bias_initializer=tf.constant_initializer(0),
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconvolutLayer(inputs, outputShape, stride=(2,2)):
    return layers.conv2d_transpose(inputs=inputs, filters=outputShape, kernel_size=4, strides=stride, 
                                    padding="same", data_format="channels_last", use_bias=True,
                                    bias_initializer=tf.constant_initializer(0),
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

def noise(size, length):
    return np.random.normal(size=(size, length))
    
def denormalize(images):
    #change to 0 -> 1 ((x + currentMin) / (currentMax - currentMin)) * (newMax - newMin) + newMin
    images = (images+1) / 2
    return images
