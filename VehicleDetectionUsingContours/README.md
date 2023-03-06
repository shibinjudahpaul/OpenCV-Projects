# Vehicle Detection using Contours
This project is an implementation of vehicle detection using contours in OpenCV. The aim of the project is to detect vehicles in an image or video simply using image processing techniques without training on any large database or special hardware requirements.
## Methodology
The project follows the following steps to detect vehicles:

* **Thresholding:** The image is converted to a binary image using thresholding to separate the foreground (vehicles) from the background.
* **Morphological operations:** An Opening process (Erosion and Dilation) is applied to the binary image to remove noise and fill gaps between foreground pixels.
* **Contour detection:** 
    1. First, Edges are detected using Canny edge detection method, which is then smoothened using the GaussianBlur method.
    2. Finally, Contours are detected in the image using the findContours function in OpenCV.
* **Contour comparison:** Each contour is compared to a set of rules to determine whether it represents a vehicle. The rules used in this project include checking the aspect ratio and size of the contour.

## Dependencies
The following dependencies are required to run this project:

* Python 3.X or higher
* OpenCV 3.X or higher
* Numpy

## Usage

The script will take an input image or video and output an image or video with bounding boxes around the detected vehicles.
Run the following command to run the code:
```
python main.py
```

## Conclusion and Improvements
This project demonstrates the use of image processing techniques in detecting vehicles using contours. 
* The project can also be extended to detect other objects such as pedestrians and bicycles.  
* The accuracy of the detection can be improved by adjusting the thresholding parameters and the contour comparison rules. 
* Most importantly, this project can be easily used in large scale as it has very minimal requirements in terms of training and hardware.



