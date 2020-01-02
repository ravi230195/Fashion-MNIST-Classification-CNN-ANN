# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:45:18 2019

@author: ravik
"""

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

x = X_train.reshape((X_train.shape[0], 28, 28,1))
x_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
x = x/255.0

y_act = keras.utils.to_categorical(y_train, num_classes=10, dtype='int')
y = y_act.reshape((y_act.shape[0],1,1, y_act.shape[1]))
model  = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (4,4), input_shape = (28,28,1) , activation = "relu"))
model.add(keras.layers.MaxPool2D(pool_size = (3,3)))
model.add(keras.layers.Conv2D(64, (4,4), activation = "relu"))
model.add(keras.layers.MaxPool2D(pool_size = (3,3)))

model.add(keras.layers.Dense(500, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
history = model.fit(x, y, epochs = 100)
y_cate = keras.utils.to_categorical(y_test, num_classes=10, dtype='int')
y_test_cat = y_cate.reshape((y_cate.shape[0],1,1, y_act.shape[1]))
loss, acc  = model.evaluate(x_test, y_test_cat , verbose=1)
print(loss, acc)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
