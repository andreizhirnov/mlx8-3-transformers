import numpy as np
from array import array
import struct

def read_images_labels(images_filepath, labels_filepath):
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i] = np.array(img) 
    
    return images, labels

def load_MNIST():
    ima_1, lab_1 = read_images_labels("data/archive/train-images.idx3-ubyte", "data/archive/train-labels.idx1-ubyte")
    ima_2, lab_2 = read_images_labels("data/archive/t10k-images.idx3-ubyte", "data/archive/t10k-labels.idx1-ubyte")
    return ima_1 + ima_2, lab_1 + lab_2

def recode_bw(x) -> np.array:
    return x.astype(np.float32) / 255.0 

def break_and_flatten(image, dims):
    rdims = image.shape
    flat = []
    for i in range(0, rdims[1], dims[1]):
        for j in range(0, rdims[0], dims[0]):
            chunk = image[j:j+dims[0],i:i+dims[1]]
            if chunk.shape[0]<dims[0] or chunk.shape[1]<dims[1]:
                temp = np.zeros(dims)
                temp[:chunk.shape[0],:chunk.shape[1]] = chunk
                chunk = temp
            flat.append(chunk.flatten())
    return np.stack(flat)

def break_and_flatten_3D(images, patch_dim):
    rdims = images.shape
    template = np.zeros((rdims[0], patch_dim, patch_dim))
    flat = []
    for i in range(0, rdims[2], patch_dim):
        for j in range(0, rdims[1], patch_dim):
            chunk = images[:,j:j+patch_dim,i:i+patch_dim]
            if chunk.shape != template.shape:
                temp = template.copy()
                temp[:,:chunk.shape[1],:chunk.shape[2]] = chunk
                chunk = temp
            flat.append(chunk.reshape(rdims[0], patch_dim**2)) 
    return np.stack(flat, axis = 1)
