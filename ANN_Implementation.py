# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:47:44 2019

@author: ravik
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import os
cwd = os.getcwd()

start = time.time()

def sigmod(y):
    return 1/(1 + np.exp(-y))

def softmax(y):
  return np.exp(y)/np.sum(np.exp(y), axis=0)

def sigmodDeravative(y):
    return y*(1-y)

def createYLabel(y):
    y_actual = np.zeros((10, y.shape[0]))
    for i in range(0,y.shape[0]):
        y_actual[y[i]][i] = 1
    return y_actual

def delta(predicted, actual):
    res = predicted - actual
    return res/actual.shape[1]

def error(pred, real):
    n = real.shape[1]
    logp = - np.log(pred[real.argmax(axis=0), np.arange(n)])
    loss = np.sum(logp)/n
    #print(loss)
    return loss

        
class NeuralNetwork():
    def __init__(self, inputSize, hidenNodes, output, alpha):
        self.inputSize = inputSize
        self.hidenNodes = hidenNodes
        self.output = output
        self.alpha = alpha
        self.b1 = np.ones((hidenNodes, 1))
        self.b2 = np.ones((output, 1))
        # normal distributuion
        self.W1 = np.random.normal(0,1, (hidenNodes, inputSize))
        self.W2 = np.random.normal(0,1,(output, hidenNodes))
        self.lossValue = []
        self.iterations = []
        
    def calculate(self, X, Y, epoch, batch_size):
        total_data_set = X.shape[0]
        print(total_data_set)
        number_of_batches = (int)(total_data_set / batch_size )
        print(number_of_batches)
        for k in range (0, epoch):
            for i in range(0, number_of_batches):
                x_batch = X[i*batch_size : i*batch_size + batch_size - 1, : ]
                y_batch = Y[ : , i*batch_size : i*batch_size + batch_size - 1]
                self.forwardProp(x_batch, y_batch)
            print(str(k)+"/"+str(epoch), end =" ")
            loss = error(self.predict(X, Y), Y)
            self.lossValue.append(loss)
            self.iterations.append(k)
            print(loss)
        
    def forwardProp(self, X, Y):
        self.x = X
        self.y = Y
        Z2 = np.matmul(self.W1, self.x.transpose()) + self.b1
        self.a2 = sigmod(Z2)
        Z3 = np.matmul(self.W2,self.a2) + self.b2
        self.a3 = softmax(Z3)
        self.backProp()


    
    def backProp(self):
        a3_delta = delta(self.a3, self.y)            # use foe dw2
        z1_delta = np.matmul(self.W2.transpose(), a3_delta)  #use for dw1
        a2_delta = z1_delta*sigmodDeravative(self.a2)
        dw2 = np.matmul(a3_delta, self.a2.transpose())
        dw1 = np.matmul(a2_delta, self.x)
        b2 = np.sum(a3_delta, axis = 1, keepdims =True)
        b1 = np.sum(a2_delta, axis = 1, keepdims = True)
        self.W1 = self.W1 - self.alpha*dw1
        self.W2 = self.W2 - self.alpha*dw2
        self.b1 = self.b1 - self.alpha*b1
        self.b2 = self.b2 - self.alpha*b2

    def predict(self, X, Y):
        Z2 = np.matmul(self.W1, X.transpose()) + self.b1
        a2 = sigmod(Z2)
        Z3 = np.matmul(self.W2, a2) + self.b2
        a3 = softmax(Z3)
        return a3
        
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

y_actual = createYLabel(y_train)
epochs = 200
NN = NeuralNetwork(X_train.shape[1],500,y_actual.shape[0], 0.1)
NN.calculate(X_train/255.0, y_actual, epochs, 5000)
plt.plot(np.arange(epochs), NN.lossValue)
plt.show()
print (time.time() - start)

y_test_labeled = createYLabel(y_test)
y_pridict = NN.predict(X_test, y_test_labeled)
pred_index = y_pridict.argmax(axis = 0)
actual_index = y_test_labeled.argmax(axis=0)

total = 0
for i in range (0, actual_index.shape[0]):
    if pred_index[i] == actual_index[i]:
        total += 1
print(total/actual_index.shape[0])

from sklearn.metrics import confusion_matrix
import seaborn as sn
cm = confusion_matrix(y_test_labeled.argmax(axis=0), y_pridict.argmax(axis=0))
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
