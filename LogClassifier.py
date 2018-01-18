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

def sigmoid(x):
    return 1 / (1 + exp( -x))

def train_log_classifier(num, X, Y, W, b):
    for i in range(num):
        y_p = get_prediction(X[i], W, b)
    return []

def get_prediction(X, W, b):
    return sigmoid(np.dot(W,X)+b)

if __name__ == "__main__":
    num, images, labels = get_training_data()

    print(images[1,:,:])
