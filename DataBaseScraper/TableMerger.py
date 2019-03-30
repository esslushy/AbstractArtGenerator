import sqlite3
import pandas as pd
import numpy as np
import urllib
import cv2

#connect
conn = sqlite3.connect("BamImages.sqlite")
#read from table
IDwithInfo = pd.read_sql(sql='select * from automatic_labels;', con=conn)
IDwithLink = pd.read_sql(sql='select * from modules;', con=conn)
print('read files')
#setup a link column
IDwithLink = IDwithLink.drop(['project_id', 'mature_content', 'license'], axis=1)
#merge them together
MergedInfo = pd.merge(IDwithInfo, IDwithLink, on='mid').drop('mid', axis=1)
#remove unused items hogging ram
del IDwithInfo
del IDwithLink
conn.close()
#check to make sure it worked
print(MergedInfo)
print(MergedInfo.isnull().values.any())
print(MergedInfo.nunique())
#put rearrange to logical order
print(MergedInfo.columns.values)
MergedInfo = MergedInfo[['src', 'content_building', 'content_flower', 'content_bicycle', 'content_people', 'content_dog', 'content_cars', 'content_cat', 'content_tree', 'content_bird',
                            'emotion_happy', 'emotion_scary', 'emotion_gloomy', 'emotion_peaceful', 'media_comic', 'media_3d_graphics', 'media_vectorart', 'media_graphite', 'media_pen_ink', 
                            'media_oilpaint', 'media_watercolor']]
print(MergedInfo.columns.values)
#change negative to 0, unsure to .5, and positive to 1
MergedInfo = MergedInfo.replace('negative', 0).replace('unsure', .5).replace('positive', 1)
print(MergedInfo)
#change it so that the last is a one hot encoding of the 20 categoires
OneHotValues = MergedInfo.iloc[:, 1:].values
SrcValues = MergedInfo.iloc[:, :1].values
#flush old dataframe
del MergedInfo
#setup new dataset



np.save('../dataset/images.npy', SrcValues)
np.save('../dataset/tags.npy', OneHotValues)
