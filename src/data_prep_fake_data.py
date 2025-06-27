import numpy as np
from src.data_prep_functions import load_MNIST, recode_bw, break_and_flatten

np.random.seed(123)
images, labs = load_MNIST()
vocab = np.array([str(i) for i in range(10)] + ["<f>"] + ["<s>","<p>"])
lookup = {w: i for i, w in enumerate(vocab)}


class Combiner:
    def __init__(self, images, labels, lookup, pr_empty):
        self.pr_empty = pr_empty
        self.images = images
        self.empty_tile = np.zeros(images[0].shape)
        self.labels = labels
        self.seq = np.arange(len(images))
        self.lookup = lookup
      
    def draw_seq(self):
        tiles = []
        labs = [lookup["<s>"]]
        e = np.random.rand(4)
        for i, u in enumerate(e):
            if u < self.pr_empty:
                tiles.append(self.empty_tile)
            else:
                loc = int(np.random.choice(self.seq, 1))
                tiles.append(self.images[loc])
                labs.append(lookup[str(self.labels[loc])])
        
        co_tiles = np.block([[tiles[0], tiles[1]],[tiles[2], tiles[3]]])
        labs.append(lookup["<f>"])
        if len(labs)<6:
            for i in range(6-len(labs)): labs.append(lookup["<p>"])
        return co_tiles, np.array(labs)

builder = Combiner(images, labs, lookup, pr_empty=0.2)
 
# ## take a look at a sample image
# xx = builder.draw_seq()
# import matplotlib.pyplot as plt 
# plt.figure(figsize=(30,20))
# image = xx[0]
# plt.imshow(image, cmap=plt.cm.gray)
# print(xx[1])

nima = []
nlabs = []
for i in range(50_000):
    im, l = builder.draw_seq()
    nima.append(recode_bw(im))
    nlabs.append(l)
nima = np.stack(nima)
nlabs = np.stack(nlabs)    

np.save(arr=nima, file="data/comb_images.npy")
np.save(arr=nlabs, file="data/comb_labs.npy")
np.save(arr=vocab, file="data/vocab.npy")

