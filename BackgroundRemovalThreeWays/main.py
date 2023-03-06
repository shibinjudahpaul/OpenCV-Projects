# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:37:38 2023

@author: Shibin Judah Paul
"""

import cv2
import numpy as np
 
from matplotlib import pyplot as plt


def bgr1(myimage):
 
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage,(5,5), 0)
 
    # We bin the pixels. Result will be a value 1..5
    bins=np.array([0,51,102,153,204,255])
    myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foregorund
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
 
    return finalimage

def bgr2(myimage):
    # First Convert to Grayscale
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    return finalimage

def bgr3(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
 
    return finalimage

# create a function to open the webcam
def open_webcam(choice):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            #switch between choices
            if meth_choice == "1":
                frame = draw_label(bgr1(frame), 'Method 1: ', (60,60), (0,0,255))
            elif meth_choice == "2":
                frame = draw_label(bgr2(frame), 'Method 2: ', (60,60), (0,0,255))
            elif meth_choice == "3":
                frame = draw_label(bgr3(frame), 'Method 3: ', (60,60), (0,0,255))
            elif meth_choice == '4':
                frame1 = draw_label(bgr1(frame), 'Method 1: ', (60,60), (0,0,255))
                frame2 = draw_label(bgr2(frame), 'Method 2: ', (60,60), (0,0,255))
                frame3 = draw_label(bgr3(frame), 'Method 3: ', (60,60), (0,0,255))
                
                frame = np.hstack((frame1, frame2, frame3))
            else:
                frame = draw_label(bgr1(frame), 'Method 1: ', (60,60), (0,0,255))
            
            cv2.imshow("frame", cv2.resize(frame,None,fx=0.4,fy=0.4))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('>>> Streamed video with method '+meth_choice+' now!')
    cap.release()
    cv2.destroyAllWindows()


def draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_DUPLEX
   scale = 1.6
   color = (0, 0, 0)
   thickness = cv2.FILLED
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   return cv2.putText(img, text, pos, font_face, scale, bg_color, 1, cv2.LINE_AA)



welcome_msg = """
\nWelcome to Background Blurring using OpenCV! 
\n=============================================
\nThis app has 3 different methods that can be used to blur the BG:
\nMethod 1: Gaussian Blur + Image Color Binning + Otsu-Thresholding 
\nMethod 2: Simple Image Thresholding to extract FG and BG 
\nMethod 3: Work in HSV colorspace + Numpy to perform Thresholding 
to extract FG and BG
\nPress 4: to compare all three in a single window
"""

print(welcome_msg)
meth_choice = input(""">>> Enter 1, 2, 3 or 4 to choose a method:\n>>> """)

ip_choice = input(">>> Enter 'i' to upload an image or 'w' to open the webcam:\n>>> ")

if ip_choice == 'i':
    # prompt the user to enter the path to the image file
    path = input(">>> Enter the path to the image file or press enter for default:\n>>> ")
    
    if (not path):
        path = 'sample/Fore-Back-Mid.jpg'
    
    # load the image from the file
    img = cv2.imread(path)
    
    #switch between choices
    if meth_choice == "1":
        img = draw_label(bgr1(img), 'Method '+meth_choice+': ', (60,60), (0,0,255))
    elif meth_choice == "2":
        img = draw_label(bgr2(img), 'Method '+meth_choice+': ', (60,60), (0,0,255))
    elif meth_choice == "3":
        img = draw_label(bgr3(img), 'Method '+meth_choice+': ', (60,60), (0,0,255))
    elif meth_choice == '4':
        img1 = draw_label(bgr1(img), 'Method 1: ', (60,60), (0,0,255))
        img2 = draw_label(bgr2(img), 'Method 2: ', (60,60), (0,0,255))
        img3 = draw_label(bgr3(img), 'Method 3: ', (60,60), (0,0,255))
        
        img = np.hstack((img1, img2, img3))
    else:
        img = draw_label(bgr1(img), 'Method '+meth_choice+': ', (60,60), (0,0,255))
    
    # do something with the image, such as display it or process it with OpenCV functions
    cv2.imshow("image", cv2.resize(img,None,fx=0.8,fy=0.8))
    cv2.imwrite('results/result.jpg', img)
    print('>>> Success! Showing image with after applying method '+ meth_choice+' now!')
    print('>>> Please find the converted Image saved in Results folder.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif ip_choice == 'w':
    # open the webcam
    open_webcam(meth_choice)

else:
    print(">>> Invalid choice.")
    