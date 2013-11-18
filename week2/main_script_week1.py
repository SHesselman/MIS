# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sgn
from PIL import Image
from scipy.ndimage import filters

import sys
sys.path.insert(0, '../')
import tools
import week1

# PREFERENCES FOR DISPLAYING ARRAYS. FEEL FREE TO CHANGE THE VALUES TO YOUR LIKING
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step A. Download images [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cd "CHANGE THIS PATH TO YOUR WORKING DIRECTORY $main_dir/python/week1"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step B. Basic image operations [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# B.1 Read image
im = array(imread('../../data/objects/flower/1.jpg'))

# B.2 Show image
imshow(im)
axis('off')

# B.3 Get image size
H, W, C = im.shape    # H for height, W for width, C for number of color channels
print 'Height: ' + str(H) + ', Width: ' + str(W) + ', Channels: ' + str(C)

# B.4 Access image pixel
print im[0, 0, 1]    # Single value in the 2 color dimension. Remember, numbering start from 0 (thus 1 means "2")
print im[0, 0]       # Vector of RGB values in all 3 color dimensions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step C. Compute image histograms [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compute color histogram from channel R
# C.1 Vectorize first the array
im_r = im[:, :, 0].flatten()

# C.2 Compute histogram from channel R using the bincount command, as indicated in the handout
histo_r = np.bincount(im_r, None, 256)

# C.3 Compute now the histograms from the other channels, that is G and B
im_g = im[:,:,1].flatten()
histo_g = np.bincount(im_g, None, 256)

im_b = im[:,:,2].flatten()
histo_b = np.bincount(im_b, None, 256)

# C.4 Concatenate histograms from R, G, B one below the other into a single histogram
histo = np.concatenate([histo_r, histo_g, histo_b])
matplotlib.pyplot.bar(range(0, len(histo)), histo, 0.8, None, None, color='red')

######
# C.5 PUT YOUR CODE INTO THE FUNCTION extractColorHistogram( im ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step D. Compute distances between vectors [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# D.1 Open images and extract their RGB histograms
im1 = imread('../../data/objects/flower/1.jpg')
histo1 = week1.extractColorHistogram(im1)
im2 = imread('../../data/objects/flower/3.jpg')
histo2 = week1.extractColorHistogram(im2)

# D.2 Compute euclidean distance: d=Σ(x-y)^2 
# Note: the ***smaller*** the value, the more similar the histograms
dist_euc = np.linalg.norm(histo2-histo1)
print dist_euc

# D.3 Compute histogram intersection distance: d=Σmin(x, y)
# Note: the ***larger*** the value, the more similar the histograms
dist_inter = np.sum(np.minimum(histo1,histo2))
print dist_inter

# D.4 Compute chi-2 similarity: d= Σ(x-y)^2 / (x+y)
# Note: the ***larger*** the value, the more similar the histograms
for i in range(0, len(histo)):
    dist_chi2 = np.sum(histo1[i]-histo2[i])**2 / (histo1[i]+histo2[i])
    print dist_chi2

# D.5 Compute hellinger distance: d= Σsqrt(x*y)
# Note: the ***larger*** the value, the more similar the histograms
for i in range(0, len(histo)):
    dist_hell = np.sum(np.sqrt(histo1[i]*histo2[i]))
    print dist_hell

######
# D.6 PUT YOUR CODE INTO THE FUNCTION computeVectorDistance( vec1, vec2, dist_type ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step E. Rank images [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# E.1 Compute histograms for all images in the dataset
impaths = tools.getImagePathsFromObjectsDataset('flower') # [ALREADY IMPLEMENTED]
histo = []

for i in list(xrange(len(impaths))):
    # print impaths[i] # get list of all flower images
    histo.append(week1.extractColorHistogram(array(imread(impaths[i]))))

# E.2 Compute distances between all images in the dataset
imdists = []
for i in range(0,60):
    for j in range(0,60):
        imdists.append([])
        imdists[i].append(week1.computeVectorDistance(histo[i], histo[j], 'chi2'))

# E.3 Given an image, rank all other images
query_id = int(14) # rnd.randint(0, 60) # get a random image for a query
sorted_id = np.argsort(imdists[query_id])
print sorted_id

# E.4 Showing results. First image is the query, the rest are the top-5 most similar images [ALREADY IMPLEMENTED]
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
im = imread(impaths[query_id])
ax.imshow(im)
ax.axis('off')
ax.set_title('Query image')

for i in np.arange(1, 1+5):
    ax = fig.add_subplot(2, 3, i+1)
    im = imread(impaths[sorted_id[i-1]]) # The 0th image is the query itself
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(impaths[sorted_id[i-1]])

######
# E.5 PUT YOUR CODE INTO THE FUNCTIONS computeImageDistances( images )
#     AND rankImages( imdists, query_id ) IN week1.py
######

# F. Gaussian blurring using gaussian filter for convolution

# F.1 Open an image
im = array(Image.open('../../data/objects/flower/1.jpg').convert('L'))
imshow(im, cmap='gray') # To show as grayscale image

# F.2 Compute gaussian filter
sigma = 10.0
half_size = 3*sigma
x = np.arange(-half_size, half_size + 1)    
G = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x**2)/(2*sigma**2))
print G 

# F.3 Apply gaussian convolution filter to the image. See the result. Compare with Python functionality
im_gf = week1.apply_gaussian_conv(im, G) # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN FILTER G]
im_gf2 = filters.gaussian_filter(im, sigma) # The result using Python functionality

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_gf, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_gf2, cmap='gray')

# F.4 Compute first order gaussian derivative filter in one dimension, row or column
dG = -(x/sigma**2)*G

# Apply first on the row dimension
im_drow = week1.apply_filter(im, dG, 'row') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]
# Apply then on the column dimension
im_dcol = week1.apply_filter(im, dG, 'col') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_dcol, cmap='gray')

# F.6 Compute the magnitude and the orientation of the gradients of an image
im_dmag = np.sqrt((im_drow**4)+(im_dcol**4))

fig = plt.figure()
imshow(im_dmag, cmap='gray')

######
# F.6.1 PUT YOUR CODE INTO THE FUNCTIONS get_gaussian_filter(sigma),
#       get_gaussian_der_filter(sigma, order) AND gradmag(im_drow, im_dcol) IN week1.py
######

# F.7 Apply gaussian filters on impulse image. HERE YOU JUST NEED TO USE THE CODE
#     YOU HAVE ALREADY IMPLEMENTED

# F.7.1 Create impulse image
imp = np.zeros([15, 15])
imp[6, 6] = 1
imshow(imp, cmap='gray')

# F.7.1 Compute gaussian filters
sigma = 1.0
G = week1.get_gaussian_filter(sigma) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED

fig = plt.figure()
plt.plot(G)
fig.suptitle('My gaussian filter') # HERE YOU SHOULD GET A BELL CURVE

# F.7.2 Apply gaussian filters
imp_gfilt = week1.apply_gaussian_conv(imp, G) # [ALREADY IMPLEMENTED, ADDED HERE ONLY FOR VISUALIZATION PURPOSES]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_gfilt, cmap='gray')
ax.set_title('Gaussian convolution: my implementation')
ax = fig.add_subplot(1, 2, 2)
imshow(tools.gf_2d(sigma, H), cmap='gray')
ax.set_title('Gaussian Kernel already provided')

# F.7.3 Apply first order derivative gradient filter
dG = week1.get_gaussian_der_filter(sigma, 1) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED
imp_drow = week1.apply_filter(imp, dG, 'row') # [ALREADY IMPLEMENTED]
imp_dcol = week1.apply_filter(imp, dG, 'col') # [ALREADY IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
imshow(imp_dcol, cmap='gray')

