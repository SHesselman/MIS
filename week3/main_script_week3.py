# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE A : GO TO MAIN DIR FOR WEEK2, THAT IS WHERE MAIN_SCRIPT_WEEK2.PY IS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.cluster.vq as cluster
import matplotlib.cm as cmx
from scipy import ndimage
from collections import defaultdict
import pickle
import math
import random
import sys
import os

import week3
sys.path.insert(0, '../')
import tools

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 1. KMEANS

# GENERATE RANDOM DATA
x, labels = week3.generate_2d_data()

week3.plot_2d_data(x, labels, None, None)

# PART 1. STEP 0. PICK RANDOM CENTERS
K = 2
means = np.array(random.sample(x, K))
week3.plot_2d_data(x, None, None, means)

# PART 1. STEP 1. CALCULATE DISTANCE FROM DATA TO CENTERS
dist = []
dist = np.zeros([K, x.shape[0]])
for i in np.arange(0, K):
    for j in np.arange(0, x.shape[0]):
        dist[i, j] = np.linalg.norm(means[i] - x[j])
        print dist[i, j]

# PART 1. STEP 2. FIND WHAT IS THE CLOSEST CENTER PER POINT
closest = np.argmin(dist, axis=0)
week3.plot_2d_data(x, None, closest, means)
print closest

# PART 1. STEP 3. UPDATE CENTERS
# for i in np.arange(0, K):
    # means[i, :] = (sum(closest[i]) / len(x[j]), sum(closest[i]) / len(x[j]))
    # print means[i, :]

for i in np.arange(0, K):
    means[i, :] = np.mean(x[closest == i, :],0)

week3.plot_2d_data(x, None, closest, means)

# means = week3.mykmeans(x, K) # return new means TODO: Recall to function.

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 2. COLOR BASED IMAGE SEGMENTATION

im = Image.open('../../data/coral.jpg')
imshow(im)
im = np.array(im)
im_flat = np.reshape(im, [im.shape[0] * im.shape[1], im.shape[2]])

N = 10000
im_flat_random = np.array(random.sample(im_flat, N))

K = 1000
[codebook, dummy] = cluster.kmeans(im_flat_random, K) # RUN SCIPY KMEANS
[indexes, dummy] = cluster.vq(im_flat, codebook) # VECTOR QUANTIZE PIXELS TO COLOR CENTERS

im_vq = codebook[indexes]
im_vq = np.reshape(im_vq, (im.shape))
im_vq = Image.fromarray(im_vq, 'RGB')

figure
subplot(1, 2, 1)
imshow(im)
subplot(1, 2, 2)
imshow(im_vq)
title('K=' + str(K))

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 3. k-MEANS AND BAG-OF-WORDS

codebook = week3.load_codebook('../../data/codebook_100.pkl')
K = codebook.shape[0]
colors = week3.get_colors(K)

files = os.listdir('../../data/oxford_scaled/')

# PART 3. STEP 1. VISUALIZE WORDS ON IMAGES
word_patches = defaultdict(list)
files_random = random.sample(files, 5)

f = 'all_souls_000057.jpg'
impath = '../../data/oxford_scaled/' + f
frames, sift = compute_sift(impath, edge_thresh = 10, peak_thresh = 5) # COMPUTE SIFT
[indexes, dummy] = ...                                                 # VECTOR QUANTIZE SIFT TO WORDS

word_patches = week3.show_words_on_image(...    # VISUALIZE WORDS
    
# PART 3. STEP 2. PLOT COLORBAR
week3.get_colorbar(colors)

# PART 3. STEP 3. PLOT WORD CONTENTS
k = 0
WN = len(word_patches[k])
figure()
suptitle('Word ' + str(k))
for i in range(WN):
    subplot(int(math.ceil(sqrt(WN))), int(math.ceil(sqrt(WN))), i+1)
    imshow(Image.fromarray(word_patches[k][i], 'RGB'))
    axis('off')

# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 5. PERFORM RETRIEVAL WITH THE BAG-OF-WORDS MODEL

# PART 5. STEP 1. LOAD BAG-OF-WORDS VECTORS FROM ../../data/bow/codebook_100/ using the week3.load_bow function

# PART 5. STEP 2. COMPUTE DISTANCE MATRIX

# PART 5. STEP 3. PERFORM RANKING SIMILAR TO WEEK 1 & 2 WITH QUERIES 'all_souls_000065.jpg', 'all_souls_0000XX.jpg', 'all_souls_0000XX.jpg'
query_id = ...
ranking = ...

# PART 5. STEP 4. COMPUTE THE PRECISION@5
files, labels, label_names = week3.get_oxford_filedata()
# ...
prec5 = week3.precision_at_N(0, gt_labels, ranking, 5)

# PART 5. STEP 4. IMPLEMENT & COMPUTE AVERAGE PRECISION




