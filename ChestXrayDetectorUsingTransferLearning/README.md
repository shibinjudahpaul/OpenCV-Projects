# COVID-19 Detection from X-rays using Machine Learning
This is a Machine Learning project that aims to detect COVID-19 from X-ray images. The goal is to build a classification model that can distinguish between healthy X-rays and those that show signs of COVID-19 infection. The model uses Convolutional Neural Networks (CNN) to extract features from the images and classify them into their respective categories.

## Dataset
The dataset used in this project is taken from [Kaggle's Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 
which contains X-ray images of COVID-19 patients as well as healthy individuals. Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. 
All chest X-ray imaging was performed as part of patients’ routine clinical care.

## Dependencies
This project requires the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pandas

You can install these dependencies using pip.
```
pip install tensorflow keras numpy matplotlib pandas
```

## Usage
To train the model, run the training.py script. The trained model will be saved in the model directory.

```
python training.py
```

To test the model on a single X-ray image, run the main.py script.

```
python predict.py path/to/image
```

## Results
The model achieves an accuracy of 98.05% on the test set. The plots and a sample test report can be found in the results directory.

## Future Work
- Improve the accuracy of the model by fine-tuning the hyperparameters and using a larger dataset.
- Develop a web application for easy access to the model for healthcare professionals.
- Explore the use of other deep learning architectures for this task.
