#STEP1 IMPORT THE MODEL
# Standard useful data processing import
import tensorflow as tf
#from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
from sklearn import ensemble,preprocessing
import keras.utils.np_utils
import argparse
import cv2
import pickle
from imutils import paths
from io import StringIO
# Visualisation imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
#import seaborn as sns
#import shap
# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split

# Keras Imports - CNN
import keras 
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers 
#from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
#from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler # import the scaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import h5py



# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# help="path to input dataset (i.e., directory of images)")
# ap.add_argument("-m", "--model", required=True,
# help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
# help="path to output label binarizer")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# help="path to output accuracy/loss plot")
# args = vars(ap.parse_args())
#IMAGE_DIMS = (32, 32, 3)
#IMAGE_DIMS = (256, 256, 3)
IMAGE_DIMS = (256,256, 1)
#IMAGE_DIMS = (512, 512, 1)
nb_batch_size = 64
n_epoch=30
height=IMAGE_DIMS[0]
width=IMAGE_DIMS[1]
depth=IMAGE_DIMS[2]
# disable eager execution
tf.compat.v1.disable_eager_execution()

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#imagePaths = sorted(list(paths.list_images("E:/Progs/Images/dataset")))
#imagePaths = sorted(list(paths.list_images("E:/Progs/archive/Images/Images")))
imagePaths = sorted(list(paths.list_images("D:/GPU3/dataset10")))

random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
    #image = cv2.imread(imagePath,1)
	image = cv2.imread(imagePath,0)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	#print('image size = ',image.shape)
	data.append(image)
	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[1]
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print('data size = ',data.shape)
print('Labels size = ',labels.shape)
# print("[INFO] data matrix: {} images ({:.2f}MB)".format(
# 	len(imagePaths), data.nbytes / (65536 * 1000.0)))
# print("[INFO] data matrix: {} images ({:.2f}MB)".format(
# 	len(imagePaths), data.nbytes / (262144 * 1000.0)))
# print("[INFO] data matrix: {} images ({:.2f}MB)".format(
# 	len(imagePaths), data.nbytes / (1024 * 1000.0)))
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = keras.utils.np_utils.to_categorical(labels)

# le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print(le_name_mapping)
# loop over each of the possible class labels and show them
for (i, label) in enumerate(le.classes_):
	print("{}. {}".format(i , label))   

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(x_train, x_test, y_train, y_test) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
print('Train size = ',x_train.shape," ",y_train.shape)
print('Test size = ',x_test.shape," ",y_test.shape)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

#Step 3 : Process the data
n_classes = len(le.classes_)
print(n_classes)

# Step4 :Create the CNN model


#im_shape = (height, width, depth)
#chanDim = -1
# if we are using "channels first", update the input shape
# and channels dimension
if K.image_data_format() == "channels_first":
	im_shape = (depth, height, width)
	#chanDim = 1
else:
    im_shape = (height, width, depth)


# model.add(Conv2D(32, kernel_size = (3, 3),  
#    activation = 'relu', input_shape = im_shape)) 
# model.add(Conv2D(64, (3, 3), activation = 'relu')) 
# model.add(MaxPooling2D(pool_size = (2, 2))) 
# model.add(Dropout(0.25))
# model.add(Flatten()) 
# model.add(Dense(128, activation = 'relu')) 
# model.add(Dropout(0.5)) 
# model.add(Dense(n_classes, activation = 'softmax'))
# print("Successfully built the CNN Model!")
# print(model.summary())

model = keras.Sequential() 
model.add(Conv2D(32, kernel_size = (7, 7),  
   activation = 'relu', input_shape = im_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (5, 5), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(512, (1, 1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten()) 
#model.add(Dropout(0.4))
model.add(Dense(1024, activation = 'relu')) 
#model.add(Dense(256, activation = 'relu')) 

#model.add(Dense(512, activation = 'relu',kernel_regularizer = regularizers.l2(0.001))) 
#model.add(Dense(64, activation = 'relu',kernel_regularizer = regularizers.l2(0.001))) 
#model.add(Dropout(0.5)) 
model.add(Dense(n_classes, activation = 'softmax'))
print("Successfully built the CNN Model!")
print(model.summary())

#Step 5 : Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy']
             )
print("Model Compilation completed!")

#Visualize the model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

mc = ModelCheckpoint('D:/GPU3/Resultcnn1/best_model5.h5', monitor='val_accuracy', mode='max', save_best_only=True)
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
#Step 6 : Train the model
import time 
tstart = time.time()
H=model.fit(x_train, y_train,
                  batch_size=nb_batch_size, epochs=n_epoch,
                  validation_data=(x_test, y_test),verbose=1, callbacks=[mc,es])

tend = time.time()
elapsed = (tend - tstart)/60
print("Model trained Successfully : Took - {} mins!".format(elapsed))

#Step 7: Evaluate the model
# train_acc = model.evaluate(x_train, y_train, verbose=0)
# print("Train Loss Value: %.2f%%" % (train_acc[0]*100))
# print("Train Accuracy Value:  %.2f%%"  % (train_acc[1]*100))
scores = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss Value: %.2f%%" % (scores[0]*100))
print("Test Accuracy Value:  %.2f%%"  % (scores[1]*100))


y_pred = model.predict(x_test, verbose=0) 
y_pred = np.argmax(y_pred, axis=1) 
y_test = np.argmax(y_test,axis = 1)
idxs1 = np.argsort(y_pred)
idxs2 = np.argsort(y_test)

#pred = pred[:, 0]
#pred = np.argmax(y_pred, axis = 1) [:5]
#y_test = np.argmax(y_test,axis = 1) [:5]

# print(pred) 
# print(y_test)
# predict probabilities for test set
# yhat_probs = model.predict(x_test, verbose=0)
# # predict crisp classes for test set
# #yhat_classes = model.predict_classes(x_test, verbose=0)
# yhat_classes = np.argmax(yhat_probs,axis=1)
# # reduce to 1d array
# yhat_probs = yhat_probs[:, 0]
# yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred )
print('Accuracy: %.2f%%' % (accuracy*100))
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred ,average="macro" )
print('Precision: %.2f%%' % (precision*100))
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred , average="macro")
print('Recall:%.2f%%' % (recall*100))
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred, average="macro")
print('F1 score: %.2f%%' % (f1*100))
report = classification_report(y_test, y_pred)
print(report)
# report.to_csv('E:/Progs/Restore/ManuscriptDataset/Resultcnnslabel/report.csv',sep = ',')

#print(classification_report(y_test, y_pred ))
# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Plot normalized confusion matrix
titles_options = [
    
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=le.classes_,
        cmap=plt.cm.Blues,
        normalize= normalize,
    )
    disp.ax_.set_title(title)
    print(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    print(disp.confusion_matrix)
    plt.savefig("D:/GPU3/Resultcnn1/confusion_matrix_norm5")

plt.show()

# Plot non-normalized confusion matrix
titles_options = [
    
    (" confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=le.classes_,
        cmap=plt.cm.Blues,
        normalize= None,
    )
    disp.ax_.set_title(title)
    print(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    print(disp.confusion_matrix)
    plt.savefig("D:/GPU3/Resultcnn1/confusion_matrix5")

plt.show()
# save the model to disk
print("[INFO] serializing network...")
model.save("D:/GPU3/Resultcnn1/modelcnns5   ", save_format="h5")
# save the single-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("D:/GPU3/Resultcnn1/labelbincnn_slabel5", "wb")
f.write(pickle.dumps(le))
f.close()

# db = h5py.File("E:/Progs/Restore/modelcnn", "r")
# i = int(db["labels"].shape[0] * 0.80)
# # generate a classification report for the model
# print("[INFO] evaluating...")
# preds = model.predict(db["features"][i:])
# print(classification_report(db["labels"][i:], preds,
# target_names=db["label_names"]))

# # compute the raw accuracy with extra precision
# acc = accuracy_score(db["labels"][i:], preds)
# print("[INFO] score: {}".format(acc))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N=n_epoch

plt.subplot(2, 1, 1)
plt.plot(np.arange(0,N),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"],label="val_acc")
plt.legend(loc='lower right')
plt.title("Training  Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")


plt.subplot(2, 1, 2)
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.legend(loc='upper right')
plt.title("Training  Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("D:/GPU3/Resultcnn1/plotcnnslabbel5")
plt.show()

