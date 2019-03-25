import numpy as np
import pandas as pd
from PIL import Image
import urllib
import cv2

data = pd.read_csv('data.csv')
print(data)
downloadedImages = []
tags = []
def addToMemMap(link):
    try:
        resp = urllib.request.urlopen(link)
        print("Downloading image from " + link)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    except urllib.error.HTTPError as e:#sometimes pages are missing or server won't respond this is ok as we will still get enough data
        print(e)
        print("Tried to access " + link + " but failed.")
        #can occur from a timeout or an actual missing image in the databse which does occur every so often.
        return
    img = cv2.resize(img, (64, 64))
    img = np.array(img)
    img = img.astype('float16')
    #normalizes and converts images to -1 -> 1 range
    np.subtract(img, np.array(127.5), out=img)
    np.divide(img, np.array(127.5), out=img)
    downloadedImages.append(img)

for _, row in data.iterrows():
    print(row.values)

np.save('C:\\Users\\evans\\datasets\\data.npy', downloadedImages)
np.save('C:\\Users\\evans\\datasets\\tags.npy', tags)