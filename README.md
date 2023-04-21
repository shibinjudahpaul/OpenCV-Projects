# Computer Vision Portfolio

Welcome to my Computer Vision portfolio! This repository contains a collection of projects I have worked with OpenCV, Numpy and Tensorflow. Each project includes a brief description, the tools and technologies used, and the outcomes achieved. The code and relevant datasets are available in each project repository.

## Table of Contents:
* **Project 1 :** [Stitching and Post-Processing Panoramas](/StitchAndPostProcess)
* **Project 2 :** [Video Stabilization Using OpticalFlow](/VideoStabilizationUsingOptialFlow)
* **Project 3 :** [COVID-19 Detection from X-rays using Transfer Learning](/ChestXrayDetectorUsingTransferLearning)
* **Project 4 :** [Vehicle Detection using Contours](/VehicleDetectionUsingContours)
* **Project 5 :** [Comparision of 3 different Background Removal methods](/BackgroundRemovalThreeWays)
* **Project 6 :** [Vehicle Detector and Counter using YOLOv8](/AutoTollBooth)




## Project 1: Stitching and Post-processing Panoramas
This is a computer vision project where I have implemented an algorithm to stitch multiple images together to create a panorama. I have used OpenCV and Python to implement the algorithm, and also included post-processing techniques such as blending and warping to improve the quality of the stitched image.

## Project 2: Video Stabilization using Optical Flow
This is a video processing project where I have implemented an algorithm to stabilize shaky videos using optical flow. The algorithm uses OpenCV to compute the optical flow between consecutive frames and then applies a smoothing filter to remove the camera shake. The resulting video is smooth and stable.

## Project 3: COVID-19 Detection from X-rays using Transfer Learning
This is a deep learning project where I have trained a convolutional neural network (CNN) to classify chest X-ray images into two categories: COVID-positive and COVID-negative. The model was built using TensorFlow and Keras and achieved an accuracy of 98% on the test set.

## Project 4: Vehicle Detection using Contours
This project is an implementation of vehicle detection using contours in OpenCV. The aim of the project is to detect vehicles in an image or video simply using image processing techniques without training on any large database or hardware requirements.

## Project 5: Comparision of 3 different Background Removal methods
This project compares three different methods for background removal using OpenCV in images, videos and live webcam streams. The methods involve manipulating the foreground and background, using various thresholding techniques, and working with both BGR and HSV color spaces. One method involves extracting the foreground and background using only numpy. 

## Project 6: Vehicle Detector and Counter using YOLOv8
This project is built to detect and count vehicles on a highway in a given video file or live video stream using YOLOv8. The program outputs a transformed video file with the individual type of detected vehicles and the total count of vehicle Inflow and Outflow from the Tolling Booth. 


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
Each project is contained in its own directory, and includes a README file with detailed instructions on how to run the project and use its functionalities.

## Conclusion
This Computer Vision Portfolio showcases my skills in image processing, computer vision, and machine learning. I have implemented various projects using popular libraries such as TensorFlow and OpenCV, and have included detailed instructions on how to use and run each project.
