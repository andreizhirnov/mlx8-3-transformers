import numpy as np 
import torch
from src.data_prep_functions import load_MNIST, recode_bw, break_and_flatten

images, labs = load_MNIST()

ima = [break_and_flatten(recode_bw(im), (4,4)) for im in images]
new_images = np.stack(ima)
new_labs = np.array(labs)

np.save(arr=new_images, file="data/images.npy")
np.save(arr=new_labs, file="data/labs.npy")

