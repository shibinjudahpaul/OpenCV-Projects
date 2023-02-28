# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:15:09 2021

@author: Shibin Judah Paul
"""

import numpy as np # linear algebra
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import os


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 15
BS = 8

DIRECTORY = 'Path to the main folder'
CATEGORIES = ["Normal", "Covid"]


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print(">>> loading images")

data = []
labels = []


for category in CATEGORIES:
    path = os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224, 224))
        
        data.append(image)
        labels.append(category)
        

data1 = data.copy()
labels1 = labels.copy()

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

#Binarize labels in a one-vs-all fashion
Label_Bi = LabelBinarizer()
labels = Label_Bi.fit_transform(labels)
labels = to_categorical(labels)
#print(labels)

#split the data
(X_train, X_test, Y_train, Y_test) = train_test_split(data,
    labels,test_size=0.20, stratify = labels, random_state = 42)

#Augmenting data: rotation_range and fill_mode
AugmentedData = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# load the VGG16 network, with imagenet
baseModel = VGG16(weights="imagenet",
                  include_top=False, input_tensor=Input(shape=(224, 224, 3)))

#construct headmodel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#aligning models
fusedModel = Model(inputs = baseModel.input, outputs=headModel)

#Since baseModel (imagenet) is a pretrained model, we skip its training in
# the first iteration by freezing the model
for layer in baseModel.layers:
    layer.trainable = False
    

#compile model
print(">>> Compiling Model")


#Using ADAM opt for gradient descent
optimized = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    
fusedModel.compile(loss="binary_crossentropy", optimizer=optimized,
	metrics=["accuracy"])

#Training headModel
print(">>> Training headModel")

#Model.fit() gets the model to fit into the given data
H = fusedModel.fit(
	AugmentedData.flow(X_train, Y_train, batch_size=BS),
	steps_per_epoch=len(X_train) // BS,
	validation_data=(X_test, Y_test),
	validation_steps=len(X_test) // BS,
	epochs=EPOCHS)


print(">>> Generating Test predictions")
predicted_Index = fusedModel.predict(X_test, batch_size=BS)


#obtain the index of the highly probable label for the respective image.
predicted_Index = np.argmax(predicted_Index, axis=1)

#Build a text report showing the main classification metrics.
print(classification_report(Y_test.argmax(axis = 1), predicted_Index, 
                            target_names= Label_Bi.classes_))

#Plotting the predictions
rows = 3
columns = 3
fig = plt.figure(figsize=(20, 20))

for m in range(1, 10):
    if str(predicted_Index[m-1]) == "0":
        text = "NORMAL"
        color = (0, 255, 0)
    elif str(predicted_Index[m-1]) == "1":
        text = "COVID"
        color = (255, 0, 0)
    img = X_test[m-1].copy()
    window_name = text
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    thickness = 2
    img = cv2.putText(img, text, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    fig.add_subplot(rows, columns, m)
    plt.imshow(img)
    plt.title("Predicted: " + text)
    plt.axis('off')
plt.show()

#computing Confusion Matrix, accuracy, sensitivity and specificity
conMatrix = confusion_matrix(Y_test.argmax(axis=1), predicted_Index)
total = sum(sum(conMatrix))
accuracy = (conMatrix[0, 0] + conMatrix[1, 1]) / total
sensitivity = conMatrix[0, 0] / (conMatrix[0, 0] + conMatrix[0, 1])
specificity = conMatrix[1, 1] / (conMatrix[1, 0] + conMatrix[1, 1])


# print the confusion matrix, accuracy, sensitivity, and specificity
print(conMatrix)
print("accuracy: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
        
# save the model to disk
print(">>> saving the trained detector model")
fusedModel.save("CovidDetector.model", save_format="h5")
    