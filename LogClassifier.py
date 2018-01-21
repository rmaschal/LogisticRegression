import numpy

from ReadMNIST import *
from ClassificationUtils import *
from math import exp, log
from random import *

def cost_function(X, Y, W, b ):
    cost = 0.0

    for i in range(len(X)):
        y_p = get_prediction(X[i],  W, b)
        cost += log_loss_function(Y[i], y_p)
        

    cost /= len(X)

    return cost

def update_weights(X, Y, W, b, learn_rate=1):
    dW = 0.0
    db = 0.0
    
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        y_p = get_prediction(x, W, b)

        dW += (y_p - y)*x
        db += (y_p - y)
    
    dW /= len(X)
    db /= len(X)

    W = W - learn_rate * dW
    b = b - learn_rate * db
    
    return W, b

def train_classifier(X, Y, iterations, learn_rate):
    num, rows, cols = X.shape

    X = X.reshape(num, rows*cols)
    W = np.random.rand(rows*cols)
    b = random()

    for i in range(iterations):
        W, b = update_weights(X, Y, W, b, learn_rate)

        if i % 2 == 0:
            cost = cost_function(X, Y, W, b)
            print("Iteration: " + str(i) + " Cost: " + str(cost))

    return W, b

def get_prediction(X, W, b):
    return sigmoid(np.dot(W,X)+b)

# Create binary labels. 'elt' labelled as 1, all else as 0
def get_binary_labels(lbls, elt):
    new_lbls = np.zeros(lbls.shape)

    for i in range(len(lbls)):
        if lbls[i] == elt:
            new_lbls[i] = 1.0
        else:
            new_lbls[i] = 0.0
    return new_lbls

def get_accuracy(X, Y, W, b):
    num_correct = 0.0
    num, rows, cols = X.shape
    X = X.reshape(num,rows*cols)
    
    for i in range(len(X)):
        y_p = get_prediction(X[i], W, b)
        
        if y_p > .5:
            y_p = 1.0
        else:
            y_p = 0.0
        
        if y_p == Y[i]:
            num_correct+=1.0

    return num_correct / len(X)


if __name__ == "__main__":
    num, images, labels = get_training_data()
    num, test_images, test_labels = get_test_data()

    # Scale image to be represented by values between 0 and 1
    # Prevents some math overflow errors
    images = images/ 255.0
    test_images = test_images / 255.0

    # classify the digit 8
    W, b = train_classifier(images,get_binary_labels(labels,1), 100, 1)    

    # Get accuracy
    acc = get_accuracy(test_images, get_binary_labels(test_labels, 1), W, b)
    print("Accuracy: " + str(acc))

