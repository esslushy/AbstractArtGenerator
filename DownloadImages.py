import urllib.request
import codecs

linkFile = codecs.open("links.txt", "r", "utf-8", errors="ignore")
linkArr = linkFile.read().split("\n")
linkFile.close()
directory = "/Volumes/Seagate Exp/ImagesForArtGenerator/"
ext = ".jpg"
randnum = 1000000
for link in linkArr[]:
    try:
        urllib.request.urlretrieve(link, directory+str(randnum)+ext)
        print("Downloading image from " + link + " to directory " + directory+str(randnum)+ext)
        randnum+=1
    except urllib.error.HTTPError as e:#sometimes pages are missing or server won't respond this is ok as we will still get enough data
        print(e)
        print("Tried to access " + link + " but failed.")
        #can occur from a timeout or an actual missing image in the databse which does occur every so often.
        pass