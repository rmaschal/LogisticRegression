import numpy

from ReadMNIST import *
from math import exp, log

TRAINING_IMAGES_PATH = 'MNIST/train-images-idx3-ubyte'
TRAINING_LABELS_PATH = 'MNIST/train-labels-idx1-ubyte'
TESTING_IMAGES_PATH = 'MNIST/t10k-images-idx3-ubyte'
TESTING_LABELS_PATH = 'MNIST/t10k-labels-idx1-ubyte'

def get_training_data():
    num1, labels = read_labels(TRAINING_LABELS_PATH)
    num2, images = read_images(TRAINING_IMAGES_PATH)
    assert num1 == num2, "Number of training images and labels differ"

    return num1, images, labels 

def get_test_data():
    num1, labels = read_labels(TESTING_LABELS_PATH)
    num2, images = read_images(TESTING_IMAGES_PATH)
    assert num1 == num2, "Number of testing images and labels differ"
    
    return num1, images, labels

def cost_function(X, Y, W, b ):
    return 0

def sigmoid(x):
    return 1 / (1 + exp( -x))

def update_weights(X, Y, W, b, learn_rate=1)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        y_p = get_prediction(x, W, b)

        dW += (y_p - i)*x
        db += (y_p - i)

    W = W - learn_rate * dW
    b = b - learn_rate * db
    
def train_classifier(X, Y, iterations, learn_rate):
    num, rows, cols = X.shape

    X = X.reshape(num, rows*cols)
    W = np.rand.(rows*cols)
    b = np.rand(1)

    for i in range(iteratoins):
        update_weights(X, Y, W, b, learn_rate)

        #compute cost function, track somehow

def get_prediction(X, W, b):
    return sigmoid(np.dot(W,X)+b)

if __name__ == "__main__":
    num, images, labels = get_training_data()

    print(images[1,:,:])
