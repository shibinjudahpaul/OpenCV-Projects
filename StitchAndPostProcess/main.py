# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:52:43 2021

@author: Shibin Judah Paul
"""

import numpy as np
import cv2
import glob
import imutils

#change folder [1,2]
image_paths = glob.glob('Images/2/*')
images = []



for image in image_paths:
    curImg = cv2.imread(image)
    curImg = cv2.resize(curImg,(0,0),None,0.3,0.3)
    images.append(curImg)
    # cv2.imshow("Image", curImg)
    # cv2.waitKey(0)


imageStitcher = cv2.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:

    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("Stitched Img", stitched_img)
    cv2.waitKey(0)
    
    
    
    """
    Post Processing
    
    """
    
    """The function copies the source image into the middle of the destination image.
    The areas to the left, to the right, above and below the copied source image will 
    be filled with extrapolated pixels"""
    
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    
    # cv2.imwrite("copyMakeBorder.png", stitched_img)
    cv2.imshow("copyMakeBorder", stitched_img)
    cv2.waitKey(0)
    
    
    """convert to GS"""
    gray = cv2.cvtColor(stitched_img,cv2.COLOR_BGR2GRAY)
    
    # cv2.imwrite("grayscaled.png", gray)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    
    """Apply binary thresholding; to help with identifying btwn the borders and main image"""
    
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    # cv2.imwrite("Thresholded.png", thresh_img)
    cv2.imshow("thresh_img", thresh_img)
    cv2.waitKey(0)
    
    
    """
    The function retrieves contours or edges from the binary image using the algorithm [Suzuki85].
    The contours are a useful tool for edge detection.
    CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only
    their end points. 
    For example, an up-right rectangular contour is encoded with 4 points.
    """
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    """
    extract only the contours and find the contour with the max area
    i.e. areaOI is the outline of the stitched image
    """
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    
    """
    create a mask wrt to the shape of the thresholded image
        -numpy zeros array in the shape of the thresh image
        -obtain bounding box coordinates for AoI
        -create a rectangular mask  with the coordinates
    
    """
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
    
    """
    make two copies of the mask
    """
    
    minRectangle = mask.copy()
    subImage = mask.copy()
    
    
    """
    while the number of zeros in subImage is greater than none;
    erode the boundaries of the minRectangle and subtract minRectangle with Thresh_img
    
    -countNonZero() The function returns the number of non-zero elements in mtx 
    -erode() erodes away the boundaries of foreground object (Always try to keep
    foreground in white). It is normally performed on binary images. 
    -subtract() subtracts the pixel values in two images and merges them 
    """
    
    while cv2.countNonZero(subImage) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        subImage = cv2.subtract(minRectangle, thresh_img)
    
        
    """
    repeat findContours, grabContours and the Max contour i.e. the stitched image
    with their border coordinates; these can be finally used to slice the Originally
    stitched image.
    
    """
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    
    # cv2.imwrite("minRectangle.png", minRectangle)
    cv2.imshow("minRectangle Image", minRectangle)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]

    cv2.imwrite("stitchedOutputProcessed.png", stitched_img)
    cv2.imshow("Stitched Image Processed", stitched_img)
    cv2.waitKey(0)
    
    
else:
    print("Images could not be stitched!")
    print("Likely not enough key points being detected!")
