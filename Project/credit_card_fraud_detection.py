# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    # in order to scale data
from sklearn.metrics import classification_report,accuracy_score


import warnings as wr
wr.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv("/kaggle/input/credit-card-cvv-1234/creditcard.csv")
data.head(5)

data.shape

data.isnull().sum()

data.info()

data.Class.values

data.Class.value_counts()


plt.figure(figsize = (6,5))
sns.countplot(data.Class, color = "orange")
plt.show()

data.hist(figsize=(30,30))
plt.show()

fraud = data[data.Class == 1]

fraud             # Each row with class = 1

non_fraud = data[data.Class == 0]

non_fraud           # Each row with class = 0

print("Shape of fraud data:", fraud.shape)
print("Shape of non-fraus data:", non_fraud.shape)

nan_fraud_balanced = non_fraud.sample(4000)

nan_fraud_balanced

balanced_data = fraud.append(nan_fraud_balanced, ignore_index = True)

balanced_data     # 492 of them Class = 1 (fraud), 492 of them Class = 0 (nan_fraud)

balanced_data.Class.value_counts()

x = balanced_data.drop("Class", axis = 1)
x                                           # dataset without Class column

y = balanced_data.Class
y

plt.figure(figsize = (6,5))
sns.countplot(y, palette="Set2")
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

xtrain.shape

xtest.shape

scaler = StandardScaler()

scaled_xtrain = scaler.fit_transform(xtrain)
scaled_xtest = scaler.fit_transform(xtest)

scaled_xtrain

type(scaled_xtrain)

scaled_xtest

type(scaled_xtest)

print(scaled_xtrain.shape)
print(scaled_xtest.shape)

print(ytrain.shape)
print(ytest.shape)

190820+93987 # Total dataset rows

scaled_xtrain3d = scaled_xtrain.reshape(scaled_xtrain.shape[0],scaled_xtrain.shape[1],1)
scaled_xtest3d = scaled_xtest.reshape(scaled_xtest.shape[0],scaled_xtest.shape[1],1)

scaled_xtrain3d.shape, scaled_xtest3d.shape


# First Layer:

cnn = Sequential()
cnn.add(Conv1D(32, 2, activation = "relu", input_shape = (30,1)))
cnn.add(Dropout(0.1))

# Second Layer:

cnn.add(BatchNormalization()) # Batch normalization is a technique for training very deep neural networks 
                               # that standardizes the inputs to a layer for each mini-batch. This 
                               # has the effect of stabilizing the learning process and dramatically
                               # reducing the number of training epochs required to train deep networks

cnn.add(Conv1D(64, 2, activation = "relu"))
cnn.add(Dropout(0.2))          # prevents over-fitting (randomly remove some neurons)

# Flattening Layer:

cnn.add(Flatten())
cnn.add(Dropout(0.4))
cnn.add(Dense(64, activation = "relu"))
cnn.add(Dropout(0.5))

# Last Layer:

cnn.add(Dense(1, activation = "sigmoid"))

cnn.summary()

from keras.utils import plot_model
plot_model(cnn)

cnn.compile(optimizer = Adam(lr=0.0001), loss = "binary_crossentropy", metrics = ["accuracy"])

history = cnn.fit(scaled_xtrain3d, ytrain, epochs = 20, validation_data=(scaled_xtest3d, ytest), verbose=1)

fig, ax1 = plt.subplots(figsize= (10, 5))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
plt.show()

fig, ax1 = plt.subplots(figsize= (10, 5))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
plt.show()

from sklearn.metrics import confusion_matrix
cnn_predictions = cnn.predict_classes(scaled_xtest3d)
confusion_matrix = confusion_matrix(ytest, cnn_predictions)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cbar = False)
plt.title("CNN Confusion Matrix")
plt.show()

accuracy_score(ytest, cnn_predictions)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(ytest, cnn_predictions)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
