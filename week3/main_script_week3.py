# -*- coding: utf-8 -*-
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
K = 7
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
for i in np.arange(0, K):
    means[i, :] = np.mean(x[closest == i, :],0)

week3.plot_2d_data(x, None, closest, means)

# throw random points and recalculate the center.
# means = week3.mykmeans(x, K, codebook) # return new means TODO: Recall to function.

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
[indexes, dummy] = cluster.vq(im_flat_random, codebook) # VECTOR QUANTIZE PIXELS TO COLOR CENTERS

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

# Run separately for computing histograms in Part 4.
f = 'all_souls_000007.jpg'
impath = '../../data/oxford_scaled/' + f
frames, sift = week3.compute_sift(impath, edge_thresh = 10, peak_thresh = 5);
[indexes, dummy] = cluster.vq(sift ,codebook);

word_patches = week3.show_words_on_image(impath, K, frames, sift, indexes, colors, word_patches)

# PART 3. STEP 2. PLOT COLORBAR
week3.get_colorbar(colors)

# PART 3. STEP 3. PLOT WORD CONTENTS
k = 4
WN = len(word_patches[k])
figure()
suptitle('Word ' + str(k))
for i in range(WN):
    subplot(int(math.ceil(sqrt(WN))), int(math.ceil(sqrt(WN))), i+1)
    imshow(Image.fromarray(word_patches[k][i], 'RGB'))
    axis('off')

# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,
bow = np.bincount(indexes, None, 256)
matplotlib.pyplot.bar(range(0, len(bow)), bow, 0.8, None, None, color='pink')

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 5. PERFORM RETRIEVAL WITH THE BAG-OF-WORDS MODEL

# PART 5. STEP 1. LOAD BAG-OF-WORDS VECTORS FROM ../../data/bow/codebook_100/ using the week3.load_bow function
files = os.listdir('../../data/bow/codebook_100/')
bows = []
for f in files:
    bows.append(week3.load_bow('../../data/bow/codebook_100/' + f))

# PART 5. STEP 2. COMPUTE DISTANCE MATRIX
dist = []
dist_type = 'intersect'
dist = np.zeros(len(files)**2).reshape((len(files),len(files)))
for i in range(0,len(files)):
    for j in range(0,len(files)):
        if dist_type == 'euclidean' or dist_type == 'l2':
            dist[i][j] = sum(((bows[i][k] - bows[j][k])**2) for k in range(len(bows[i])))
            tools.normalizeL2(dist[i][j])
        elif dist_type == 'intersect' or dist_type == 'l1':
            dist[i][j] = sum((np.minimum(bows[i][k], bows[j][k])) for k in range(len(bows[i])))
            tools.normalizeL1(dist[i][j])
        elif dist_type == 'chi2':
            dist[i][j] = sum((((bows[i][k] - bows[j][k]) **2) / (bows[i][k]+bows[j][k])) for k in range(len(bows[i])))
        elif dist_type == 'hellinger':
            dist[i][j] = sum((np.sqrt((bows[i][k]*bows[j][k]))) for k in range(len(bows[i])))
            tools.normalizeL2(dist[i][j])

print dist[i][j]

# PART 5. STEP 3. PERFORM RANKING SIMILAR TO WEEK 1 & 2 WITH QUERIES 'all_souls_000065.jpg', 'all_souls_0000XX.jpg', 'all_souls_0000XX.jpg'
query_id = int(89) #89, 21, 48
ranking = np.argsort(dist[query_id])

if dist_type == 'euclidean' or dist_type == 'intersect':
    ranking = np.argsort(dist[query_id])
elif dist_type == 'chi2' or dist_type == 'hellinger':
    ranking = np.argsort(dist[query_id])[::-1]
print ranking

# PART 5. STEP 4. IMPLEMENT & COMPUTE AVERAGE PRECISION
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
im = imread(('../../data/oxford_scaled/' + files[ranking[i-1]]).replace('', '')[:-4])
ax.imshow(im)
ax.axis('off')
ax.set_title('Query image')

for i in np.arange(1, 1+5):
    ax = fig.add_subplot(2, 3, i+1)
    im = imread(('../../data/oxford_scaled/' + files[ranking[i-1]]).replace('', '')[:-4]) # The 0th image is the query itself
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(files[ranking[i-1]])

# PART 5. STEP 4. COMPUTE THE PRECISION@5 & @10
files, labels, label_names = week3.get_oxford_filedata()
# prec5 = week3.precision_at_N(query_id, labels, ranking, 5)
# print prec5
prec10 = week3.precision_at_N(query_id, labels, ranking, 10)
print prec10

# Represent images with line graphs by inserting precision values.
plt.plot([10,50,100,500,1100], [0.2,0.2,0.6,1.0,1.0], 'red') #Euclidean 5 - souls0065
plt.plot([10,50,100,500,1100], [0.6,1.0,1.0,1.0,1.0], 'green') #Euclidean 5 - radclife 390
plt.plot([10,50,100,500,1100], [0.4,0.8,0.8,0.8,0.8], 'blue') #Euclidean 5 - church 190
plt.axis([0, 1000, 0, 1.1])
plt.show()

plt.plot([10,50,100,500,1100], [0.1,0.2,0.3,0.3,0.2], 'red') #Euclidean 10 - souls0065
plt.plot([10,50,100,500,1100], [0.7,0.7,0.8,0.8,0.8], 'green') #Euclidean 10 - radclife 390
plt.plot([10,50,100,500,1100], [0.4,0.4,0.4,0.4,0.6], 'blue') #Euclidean 10 - church 190
plt.axis([0, 1000, -0.1, 1.1])
plt.show()

plt.plot([10,50,100,500,1100], [0.0,0.2,0.2,0.2,0.2], 'red') #Hellinger 5 - souls0065
plt.plot([10,50,100,500,1100], [0.0,0.0,0.0,0.0,0.0], 'green') #Hellinger 5 - radclife 390
plt.plot([10,50,100,500,1100], [0.0,0.0,0.0,0.0,0.0], 'blue') #Hellinger 5 - church 190
plt.axis([0, 1000, -0.1, 1.1])
plt.show()


plt.plot([10,50,100,500,1100], [0.1,0.1,0.1,0.2,0.3], 'red') #Hellinger 10 - souls0065
plt.plot([10,50,100,500,1100], [0.1,0.1,0.1,0.1,0.2], 'green') #Hellinger 10 - radclife 390
plt.plot([10,50,100,500,1100], [0.0,0.0,0.0,0.1,0.1], 'blue') #Hellinger 10 - church 190
plt.axis([0, 1000, -0.1, 1.1])
plt.show()

# Compute Representation histograms
filesHisto = os.listdir('../../data/oxford_scaled/')
histograms = []

histoDists = []
histoDists = np.zeros(len(filesHisto)**2).reshape((len(filesHisto),len(filesHisto)))


for f in filesHisto:
    im = array(imread('../../data/oxford_scaled/' + f))
    histograms.append(week3.extractColorHistogram(im))

for i in range(0,len(filesHisto)):
    for j in range(0,len(filesHisto)):
        histoDists[i][j] = sum(((histograms[i][k] - histograms[j][k])**2) for k in range(len(histograms[i])))

rankingHisto = np.argsort(histoDists[query_id,:])

files2, labels, label_names = week3.get_oxford_filedata()
# ...
prec5 = week3.precision_at_N(query_id, labels, rankingHisto, 5)
prec10 = week3.precision_at_N(query_id, labels, rankingHisto, 10)

print prec5, prec10