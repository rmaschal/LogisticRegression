import struct
import numpy as np

TRAINING_IMAGES_PATH = 'MNIST/train-images-idx3-ubyte'
TRAINING_LABELS_PATH = 'MNIST/train-labels-idx1-ubyte'
TESTING_IMAGES_PATH = 'MNIST/t10k-images-idx3-ubyte'
TESTING_LABELS_PATH = 'MNIST/t10k-labels-idx1-ubyte'

def read_labels(path):
    with open(path, 'rb') as f_labels:
        magic, num = struct.unpack(">II", f_labels.read(8))
        labels = np.fromfile(f_labels, dtype=np.int8)

    return num, labels

def read_images(path):
    with open(path, 'rb') as f_images:
        magic, num, rows, cols = struct.unpack(">IIII", f_images.read(16))
        images = np.fromfile(f_images, dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        
    return num, images

def label_to_binary_vector(lbl):
    vec = np.zeros(10)
    vec[lbl] = 1.0

    return vec

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
