import numpy as np

from ReadMNIST import *
from ClassificationUtils import *
from random import *

#
# SImple 2-layer neural network for k-class classification
#
def get_prediction(x, W, b):
    vsigmoid = np.vectorize(sigmoid)
    
    return vsigmoid(np.dot(np.transpose(W),x)+b)

# Assumes classes are labelled as 0 -- n-1
def get_class_prediction(y_p):
    max_ind = 0

    for i in range(len(y_p)):
        if y_p[i] > y_p[max_ind]:
            max_ind = i

    return max_ind

def cost_function(X, Y, W, b):
    cost = 0.0

    for i in range(len(X)):
        y_p = get_prediction(X[i], W, b)
        cost += log_loss_function(Y[i], y_p)

    cost /= len(X)

    return cost

def get_accuracy(X, Y, W, b):
    num_correct = 0.0
    num, rows, cols = X.shape
    X = X.reshape(num, rows*cols)

    for i in range(len(X)):
        y_p = get_prediction(X[i], W, b)
        y_p = get_class_prediction(y_p)

        if y_p == Y[i]:
            num_correct += 1.0

    return num_correct / len(X)

def update_weights(X, Y, W, b, learn_rate=1):
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        y_p = get_prediction(x, W, b)
        
        # There should be a better way to do this
        for j in range(len(y_p)):
            dW[:, j] += (y_p[j] - y[j]) * x
            db[j] += (y_p[j] - y[j])

    dW /= len(X)
    db /= len(X)

    W = W - learn_rate * dW
    b = b - learn_rate * db

    return W, b

def train_classifier(X, Y, iterations, learn_rate, num_classes):
    num, rows, cols = X.shape

    X = X.reshape(num, rows*cols)
    W = np.random.rand(rows*cols, num_classes)
    b = np.random.rand(num_classes)

    for i in range(iterations):
        W, b = update_weights(X, Y, W, b, learn_rate)
        
        cost = np.sum(cost_function(X, Y, W, b))
        print("Iteration: " + str(i+1) + "  Cost: " + str(cost))
    
    return W, b

def get_accuracy(X, Y, W, b):
    num_correct = 0.0
    num, rows, cols = X.shape
    X = X.reshape(num, rows*cols)

    for i in range(len(X)):
        y_p = get_prediction(X[i], W, b)
        y_p = get_class_prediction(y_p)

        if Y[i] == y_p:
            num_correct+=1

    return num_correct / len(X)
if __name__ == "__main__":
    num, images, labels = get_training_data()
    num, test_images, test_labels = get_test_data()

    blabels = label_to_binary_vector(labels, 10)
    images = images / 255.0
    test_images = test_images / 255.0

    W, b = train_classifier(images, blabels, 10, 1, 10)

    acc = get_accuracy(test_images, test_labels, W, b)
    print("Accuracy: " + str(acc))
