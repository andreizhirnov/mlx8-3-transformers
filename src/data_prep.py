import numpy as np 
import struct
import torch


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

## read data
ima_1, lab_1 = read_images_labels("data/archive/train-images.idx3-ubyte", "data/archive/train-labels.idx1-ubyte")
ima_2, lab_2 = read_images_labels("data/archive/t10k-images.idx3-ubyte", "data/archive/t10k-labels.idx1-ubyte")
  
ima = [break_and_flatten(recode_bw(im), (4,4)) for im in ima_1 + ima_2]
images = np.stack(ima)
labs = np.array(lab_1+lab_2)

np.save(arr=images, file="data/images.npy")
np.save(arr=labs, file="data/labs.npy")

