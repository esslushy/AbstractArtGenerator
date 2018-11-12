#imports
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile

#variables
directory = "/Volumes/Seagate Exp/ImagesForArtGenerator/"
finishedDirectory = "/Volumes/Seagate Exp/ReadyImagesForArtGenerator/"
randnum = 1000000
ext = ".jpg"

#code
allImages = [image for image in listdir(directory) if isfile(join(directory, image))]#gets all images from images folder
for imageDir in allImages:
    if (imageDir[:2] == "._"):#sometimes the listdir will pick up meta data thats not needed
        continue
    img = cv2.imread(directory + imageDir)
    readyImage = cv2.resize(img, (256, 256))#resizes to 256x256
    print(join(finishedDirectory, str(randnum) + ext))
    cv2.imwrite(join(finishedDirectory, str(randnum) + ext), readyImage)
    randnum+=1

