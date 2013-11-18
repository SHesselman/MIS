# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import numpy as np
import urllib
import os
import sys
import math
import scipy.signal as sgn

def extractColorHistogram( im ):
    # PRE [DO NOT TOUCH]
    histo = []

    # WRITE YOUR CODE HERE
    im_r = im[:,:,0].flatten()
    histo_r = np.bincount(im_r, None, 256)
    im_g = im[:,:,1].flatten()
    histo_g = np.bincount(im_g, None, 256)
    im_b = im[:,:,2].flatten()
    histo_b = np.bincount(im_b, None, 256)

    # RETURN [DO NOT TOUCH]
    histo = np.concatenate([histo_r, histo_g, histo_b])
    return histo

def computeVectorDistance( vec1, vec2, dist_type ):
    # PRE [DO NOT TOUCH]
    dist = []

    # WRITE YOUR CODE HERE
    if dist_type == 'euclidean' or dist_type == 'l2':
        dist = np.linalg.norm(vec2-vec1)
    elif dist_type == 'intersect' or dist_type == 'l1':
        dist = np.sum(np.minimum(vec1,vec2))
    elif dist_type == 'chi2':
        dist = np.sum((vec1-vec2)**2 / (vec1+vec2))
    elif dist_type == 'hellinger':
        dist = np.sum(np.sqrt(vec1)*np.sqrt(vec1))
                
    # RETURN [DO NOT TOUCH]
    return dist

def computeImageDistances( images ):
    # PRE [DO NOT TOUCH]
    imdists = []
    
    for i in list(xrange(len(impaths))):
    # print impaths[i] # get list of all flower images
        histo.append(extractColorHistogram(imread(impaths[i])))

    # E.2 Compute distances between all images in the dataset
    for j in range(0,60):
        for k in range(0,60):
            imdists[j].append(computeVectorDistance(images[j], images[k], 'euclidian'))

    # RETURN [DO NOT TOUCH]
    return imdists

def rankImages( imdists, query_id ):
    # PRE [DO NOT TOUCH]
    ranking = []

    # E.3 Given an image, rank all other images
    query_id = rnd.randint(0, 59) # get a random image for a query
    ranking = np.argsort(imdists[query_id])
    print ranking

    # E.4 Showing results. First image is the query, the rest are the top-5 most similar images [ALREADY IMPLEMENTED]
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    im = imread(impaths[queryId])
    ax.imshow(im)
    ax.axis('off')
    ax.set_title('Query image')

    for i in np.arange(1, 1+5):
        ax = fig.add_subplot(2, 3, i+1)
        im = imread(impaths[ranking[i-1]]) # The 0th image is the query itself
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(impaths[ranking[i-1]])
    
    # RETURN [DO NOT TOUCH]
    return ranking

def get_gaussian_filter(sigma):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    G = []
    
    # WRITE YOUR CODE HERE FOR DEFINING THE HALF SIZE OF THE FILTER
    half_size = 3*sigma
    x = np.arange(-half_size, half_size + 1)        

    # WRITE YOUR CODE HERE
    G = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x**2)/(2*sigma**2))
                        
    # RETURN [DO NOT TOUCH]
    G = G / sum(G) # It is important to normalize with the total sum of G
    return G
    
def get_gaussian_der_filter(sigma, order):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    dG = []
    
    # WRITE YOUR CODE HERE
    half_size = 3*sigma
    #
    x = np.arange(-half_size, half_size + 1)
    
    if order == 1:
        # WRITE YOUR CODE HERE
        dG = ((3*sigma)/((3*sigma)*x))*G
    elif order == 2:
        # WRITE YOUR CODE HERE
        dG = -(x/sigma**2)*G

    # RETURN [DO NOT TOUCH]
    return dG

def gradmag(im_dr, im_dc):
    # PRE [DO NOT TOUCH]
    im_dmag = []

    # WRITE YOUR CODE HERE
    im_dmag = np.sqrt((im_drow**4)+(im_dcol**4))
    #

    # RETURN [DO NOT TOUCH]
    return im_dmag    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def apply_filter(im, myfilter, dim):
    H, W = im.shape
    if dim == 'col':
        im_filt = sgn.convolve(im.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [H, W])
    elif dim == 'row':
        im_filt = sgn.convolve(im.T.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [W, H]).T
    
    return im_filt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_gaussian_conv(im, G):
    im_gfilt = apply_filter(im, G, 'col')
    im_gfilt = apply_filter(im_gfilt, G, 'row')
    
    return im_gfilt


        
        
    
