import struct
import numpy as np

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
