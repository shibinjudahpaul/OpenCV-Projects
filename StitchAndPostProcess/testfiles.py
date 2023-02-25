import os

import numpy as np
import cv2
import glob
import imutils

targetFolder = 'Images'
liFolders = os.listdir(targetFolder)
print(liFolders)
for f in liFolders:
    image_paths = glob.glob(targetFolder+'/' + f + '\*.jpg')
    print(image_paths)
    for i in image_paths:
        print(i)