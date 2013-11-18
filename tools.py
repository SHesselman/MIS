# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import urllib
import os
import platform
import numpy as np
import zipfile
import math
import matplotlib.pyplot as plt
from PIL import Image

def normalizeL1(vectors, my_axis=0):
    vectors = vectors.astype(float)
    sum_l1 = np.sum(vectors, my_axis)
    if my_axis == 0:
        vectors = vectors / sum_l1
    elif my_axis == 1:
        vectors = vectors / sum_l1[:, np.newaxis]
        
    return vectors    
        
def normalizeL2(vectors, my_axis=0):
    vectors = vectors.astype(float)
    sum_l2 = np.sqrt(np.sum(vectors ** 2, my_axis))
    if my_axis == 0:
        vectors = vectors / sum_l2
    elif my_axis == 1:
        vectors = vectors / sum_l2[:, np.newaxis]
        
    return vectors
        
def fileparts(file_path):
    filpat = os.path.abspath(file_path)
    filpat, dummy = os.path.split(filpat)
    filnam = os.path.basename(file_path)
    filnam, filext = os.path.splitext(filnam)
    return filpat, filnam, filext

def downloadObjectsDataset():
    url = 'http://staff.science.uva.nl/~gavves/files/objects.zip'
    print 'Downloading data from server. Please wait a bit...'
    urllib.urlretrieve(url, '../../data/objects.zip')
    print 'Data have been downloaded. Extracting them in "../../data/objects"'
    if not os.path.exists('../../data/objects'):
        os.mkdir('../../data/objects')
        
    zfile = zipfile.ZipFile('../../data/objects.zip').extractall('../../data/')
    os.remove('../../data/objects.zip')
        
def getImagePathsFromObjectsDataset( obj ):
    impaths = []
    for i in np.arange(1, 61):
        impaths.append('../../data/objects/' + obj + '/' + str(i) + '.jpg')
    
    return impaths
    
def gf_2d(sigma):
    sigma = float(sigma)
    acme = np.ceil(3 * sigma) # We want the filter to have a size at least 3σ
        
    x = np.arange(-acme, acme + 1)
    y = x
    xx, yy = np.meshgrid(x, y)
    rr = xx*xx + yy*yy
    G = 1/(2*math.pi*sigma)*np.exp(-1/(2*sigma*sigma) * rr)
    G = G / sum(G.flatten())
    return G   
    