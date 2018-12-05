#imports
import sqlite3
import pandas as pd
#variables
#note that the dataset used is the full bam dataset
conn = sqlite3.connect("ImageLinksDB.sqlite")#connect to database
IDwithInfo = pd.read_sql(sql="select * from automatic_labels;", con=conn)#import into a pandas database
MidNumbers = []
# print(IDwithInfo)
#The database is imported correctly and is ready to be sorted
IDwithInfo.apply(lambda x: MidNumbers.append(x["mid"]) if x["media_watercolor"] == "positive" else x, axis=1)#1 to apply to rows
#gets all the mid numbers to be used later when extracting the links from modules
#MidNumbers now has all the ids of the images with watercolor
print(len(MidNumbers))#if an assertion error is returned after this, data has been either been lost or gained
assert(len(MidNumbers) == 103798)#if not some are missing or extras have been gained from the data used.
IDwithImageSRC = pd.read_sql(sql="select * from modules;", con=conn)#gets table with sources
conn.close()#closes connection to reduce lag
f = open("links.txt", "a")#writing ot link file
IDwithImageSRC.apply(lambda x: f.write(x["src"] + "\n") if x["mid"] in MidNumbers else x, axis=1)#write links to links.txt to be used to download photos
