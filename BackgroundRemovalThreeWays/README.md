# Comparision of 3 different Background Removal methods
This project compares three different methods for background removal using OpenCV. The goal is to explore different techniques for separating Foreground (FG) and Background (BG) in images, manipulating the background, and then merging the FG and BG back together. The three methods that are compared are:
* **Method 1:**  Gaussian Blur + Binning + Otsu Thresholding
  - Perform Gaussian Blur to remove noise.
  - Simplify the image by binning the pixels into six equally spaced bins in RGB space.
  - Convert the image into greyscale and apply Otsu thresholding to obtain a mask of the FG.
  - Apply the mask onto the binned image keeping only the FG - thereby removing the BG.

* **Method 2:** Greyscale + Simple Thresholding using OpenCV
  - Convert the image into Greyscale.
  - Perform Simple thresholding to build a mask for the FG and BG.
  - Determine the FG and BG based on the mask.
  - Reconstruct original image by combining FG and BG.

* **Method 3:** HSV + Simple Thresholding using Numpy
  - Convert the image into HSV color space.
  - Perform simple thresholding to create a map using Numpy based on Saturation and Value.
  - Combine the map from S and V into a final mask.
  - Determine the FG and BG based on the combined mask.
  - Reconstruct original image by combining the extracted FG and BG.
  
## Dependencies
  The projects in this repository use the following dependencies:

  - Python 3.x
  - TensorFlow 2.x
  - OpenCV
  - NumPy
  - Matplotlib
  
  You can install these dependencies using pip.
```
pip install tensorflow opencv-python numpy matplotlib
```

## Usage 
Run the following command to run the code:
```
python main.py
```

Once the program is running, a brief of each method will be given and a choice will be required to move forward. 
```
Welcome to Background Blurring using OpenCV! 
=============================================
This app has 3 different methods that can be used to blur the BG:
Method 1: Gaussian Blur + Image Color Binning + Otsu-Thresholding 
Method 2: Simple Image Thresholding to extract FG and BG 
Method 3: Work in HSV colorspace + Numpy to perform Thresholding to extract FG and BG
Press 4: to compare all three in a single window

>>> Enter 1, 2, 3 or 4 to choose a method:
>>> |
```

Then the following will be asked:
```
>>> Enter 'i' to upload an image or 'w' to open the webcam:
>>> |
```
In case of an image, the following will be asked:
```
>>> Enter the path to the image file or press enter for default:
>>> |
```
The image result will be saved in the [Results](/results) folder. In case of webcam, a window will popup with the webcam stream in the chosen filter.

## Conclusion and Future Improvements
- BGR2 performed better overall as the mid-point method using simple thresholding produced the best image clarity and preserved a lot of material details. However, it works best with a white background and depends on simple thresholding on a greyscale image.

- Gaussian blur and color binning reduce image fidelity at high processing cost, best suited for niche applications. BGR1 can be used as a pre-processing step when image or memory size is a priority.

- BGR3 using HSV space did the best in preserving color contrast, especially for shiny objects like the top of a can. Images with a high contrasting background may improve threshold values.

- Performance can be improved further by leveraging GPUs.
