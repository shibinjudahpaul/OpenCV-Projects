# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:16:29 2021

@author: Shibin Judah Paul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_DUPLEX
   scale = 3
   color = (0, 0, 0)
   thickness = cv2.FILLED
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   return cv2.putText(img, text, pos, font_face, scale, bg_color, 5, cv2.LINE_AA)


vid = cv2.VideoCapture('samples/Traffic1.mp4')
#vid = cv2.VideoCapture(0)
kernel = np.ones((3,3),np.uint8)
#kernel = None

"""
creates the background object; 
history - the no. of frames needed before BG is decided.
varThresh - thresholding var used to filter pixels and decide on FG;
            too less - noisy and too high - flat; default is 16
detectShadows - bool decides whether to detect and mark shadows in grey.
"""   
#backgroundObj = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
backgroundObj = cv2.createBackgroundSubtractorKNN(detectShadows=True)

while True:
    check, frame = vid.read()
    
    if not check:
        break
        
        
    """Apply the BG object to every frame like a mask to extract FG"""   
    fgMask = backgroundObj.apply(frame)
    
    
    """
    thresholding the mask to get rid of overlapping shadows
    """
    _,fgmask = cv2.threshold(fgMask, 250, 255,cv2.THRESH_BINARY)
    
    
    """
    Opening operation:
    Eroding - Erodes away the boundaries of the foreground object.
            - Used to diminish the features of an image(lesser noise)
    Dilation - Increases the object area
             - Used to accentuate features
    """
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    fgMask = cv2.dilate(fgMask, kernel, iterations = 3)
    fgmask = cv2.Canny(fgmask,100,200)
    fgMask = cv2.GaussianBlur(fgMask, (7, 7), 0)
    """
    Find the contours of the cars in the images
    """
    contours,_ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colorCopy = frame.copy()
    
    for cont in contours:
        contArea = cv2.contourArea(cont)
        #print(contArea)
        if contArea > 1500 and contArea < 30000:
            
            #Get Bounding Box coords
            x,y,w,h = cv2.boundingRect(cont)
            
            #draw a BB ; 
            cv2.rectangle(colorCopy, (x,y), (x+w, y+h), (0,0,255), 2)
            
            #label the detection below BB
            cv2.putText(colorCopy, 'Vehicle' + str(contArea),(x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, cv2.LINE_AA)
            
    #Obtaining FG using the mask
    fgLayer = cv2.bitwise_and(frame,frame,mask=fgMask)
    
    #stacking the OG Frame, extracted FG frame and Result frame
    #finalOP = np.hstack((frame, fgLayer, colorCopy))
    
    #adding labels
    fgLayer = draw_label(fgLayer, 'Thresholding Contours',(60,60), (255,255,255))
    colorCopy = draw_label(colorCopy, 'Vehicle Detection',(60,60), (255,255,255))

    #stack the output horizontally
    finalOP = np.hstack((fgLayer, colorCopy))
    
    #results
    cv2.imshow('All frames',cv2.resize(finalOP,None,fx=0.4,fy=0.4))
    #cv2.imshow('results', colorCopy)
    
    k = cv2.waitKey(1) &  0xff
    if k == 27:
        break
   
vid.release()
cv2.destroyAllWindows()