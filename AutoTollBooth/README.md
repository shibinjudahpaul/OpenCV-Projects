# Vehicle Detector and Counter using YOLOv8
This project is built to detect and count vehicles in a given video file or live video stream of a highway using YOLOv8. The program outputs a video file with the detected vehicles and the total count of vehicle Inflow and Outflow from the Tolling Booth. This project is also available as a [Kaggle Notebook](https://www.kaggle.com/code/shibinjudah/automatictollbooth#Output-Video). 

# Methodology

1. Create a video object using OpenCV and read the video frame by frame.
2. Scale down the video by a scaling factor for better performance. Default value is 50%.
3. Use the YOLO model to predict the desired label IDs with a confidence value (preferably 70% or above).
4. Extract bounding box coordinates, confidence values and class IDs from the prediction.
5. Use the obtained information to draw bounding boxes along with their respective center points, and print the class and confidence values to each detected object.
6. To identify traffic Inflow or Outflow:
    * Draw a reference line at 3/4th of the frame. 
    * This reference line plus an offset with be the region of interest to determine vehicle Inflow or Outlfow.
    * Calculate the distance between the center of the bounding boxes and the reference line. Based on the polarity of the resulting distance, the overall inflow or the outflow count is recorded. 
7. The count of individual types of inbound or outbound vehicles are also maintained using the above technique along with a dictionary of labels and their respective count on either side of the road.

# Dependencies

* OpenCV
* Ultralytics
* Yolo V8
* Matplotlib
* Seaborn
* Numpy
* Pandas
* Ipython

# Usage
The notebook is divided into multiple sections and is well documented with necessary comments. Do check out the Kaggle notebook for the most recent version and dont forget to upvote if you found the project interesting!

# Results
![Transformed Output Frame](https://github.com/shibinjudahpaul/OpenCV-Projects/blob/master/AutoTollBooth/results/final_result.jpg?raw=true)

[Transformed Output Video](results/predicted_result.mp4)

# Future improvements
* Add total toll charges earned on each side of the road.
* Add speed estimation.
* Add pollution estimation.
* Add most popular type of vehicle on each side of the road.

# Conclusion
Yolo V8 is here! This was a simple project to get my hands dirty with the new model. As the current SOTA model, YOLOv8 builds on the success of previous versions, introducing new features and improvements for enhanced performance, flexibility, and efficiency. It supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification. This versatility is very exciting and I am looking forward to using it in more tasks. 
