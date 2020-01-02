# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:48:37 2019

@author: ravik
"""

# Read Fashion MNIST dataset

import keras
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
#from livelossplot import PlotLossesKeras

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist(cwd, kind='train')
X_test, y_test = load_mnist(cwd, kind='t10k')


print(X_train.shape)
y_act = keras.utils.to_categorical(y_train, num_classes=10, dtype='int')
print(y_act.shape)
# Your code goes here . . .
model = keras.models.Sequential()
model.add(keras.layers.Dense(500, activation = "sigmoid", bias_initializer='ones',input_dim=X_train.shape[1]))
model.add(keras.layers.Dense(50, activation = "sigmoid", bias_initializer='ones'))
model.add(keras.layers.Dense(10, activation = "softmax", bias_initializer='ones'))
model.compile(optimizer = 'SGD', loss = "categorical_crossentropy", metrics = ['accuracy'])
history = model.fit(X_train, y_act, epochs = 50)
loss, acc  = model.evaluate(X_test, keras.utils.to_categorical(y_test, num_classes=10, dtype='int'), verbose=1)
print(loss, acc)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sn
y_pre = model.predict(X_test)
cm = confusion_matrix(y_pre.argmax(axis=1), keras.utils.to_categorical(y_test, num_classes=10, dtype='int').argmax(axis=1))
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
