import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clr
#import scipy as sc
#import scipy.cluster.vq as cluster
import random
import os
import matplotlib.cm as cmx
import pickle
from collections import defaultdict
import math
sys.path.insert(0, '../')
import tools
import math
import pylab as pl
from sklearn import svm, datasets

import week56

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

####

files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

C = 100
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

#*****
# Q1: IMPLEMENT HERE kNN CLASSIFIER. 
#*****
# YOU CAN USE CODE FROM PREVIOUS WEEK

# Compute the distance matrix. dist =  len(testset) x len(trainset)
dist = np.zeros([len(testset),len(trainset)])
for i in range(len(testset)):
    b_i = tools.normalizeL1(bow[testset[i],:])
    for j in range(len(trainset)):
        b_j = tools.normalizeL1(bow[trainset[j],:])
        dist[i,j] = sum(np.minimum(b_i,b_j))

# Rank all images
ranking = np.argsort(dist,1)
ranking = ranking[:,::-1]

# The lines above have computed and sorted the distances for all test images
# Below, for a single test image we will visualize the results

# Get query id in {0 ... 599}, and transform it to the {0... 99} range
query_id = 366# (366, goal), (150, bicycle), (84, beach ), (450, mountain)
qinx = find(testset==query_id)


#Define K
K = 9 

# Find the nearest labels
nearest_labels = labels[trainset[ranking[qinx,0 : K]]]

# Visualize the results
figure
subplot(2, 6, 1)
imshow(Image.open(files[query_id]))
title('Query')
axis('off')

for cnt in range(K):
    subplot(2, 6, cnt+2)
    imshow(Image.open(files[int(trainset[ranking[qinx,cnt]])]))
    title(unique_labels[nearest_labels[0,cnt]-1])
    axis('off')

# Find the label for this query
# count for each possible label
# NOTE: LABELS is in 1...10, while python uses 0 ... 9
lCount = np.random.randint(0,10,len(unique_labels)) # this is a random filling, different possibilities to fill correctly

# Find the most frequent label
qLabel = 3 #This is random, fill it correctly
qLText = unique_labels[qLabel]

print "%4d | %4d | gt label %10s |" %(query_id,qinx,unique_labels[labels[query_id]-1])
print "Label occurrences: ",; print lCount
print "Predicted label: %4d %10s" %(qLabel,qLText)



# Q2: USE DIFFERENT STRATEGY

#*****
# Q3: For K = 9, COMPUTE THE CLASS ACCURACY FOR THE TESTSET
#*****

#predict labels for each image in the testset
K = 9
pred = np.zeros(len(testset))
for i in range(len(testset)):
    print "%3d" %(i)
    i_nearest_labels = labels[trainset[ranking[0 : K]]]
    print "\t",; print i_nearest_labels
    i_label_count = np.bincount(0,4,len(unique_labels)) # this is a random filling, different possibilities to fill correctly
    print "\t",; print i_label_count
    pred[i] = 3 # this is random, fill in correctly

labelstest = labels[testset]
classAcc = np.zeros(len(unique_labels))
for c in range(len(unique_labels)):
    # Find the true positives, that is the number of images for which pred == labelstest and labelstest == c
    tp = ...
    # Find the false negatives, that is the number of images for which pred != labelstest and labelstest == c
    fn = ...
    classAcc[c] = tp / (tp + fn + 0.0)

# REPORT THE CLASS ACC *PER CLASS* and the MEAN
# THE MEAN SHOULD BE (CLOSE TO): 0.31

#*****
# Q4: DO CROSS VALIDATION TO DEFINE PARAMETER K 
#*****
K = [1, 3, 5, 7, 9, 15]

# - SPLIT TRAINING SET INTO THREE PARTS.
# Use random indices and three groups of (almost) the same size
inx1 = trainset[0:150]
inx2 = trainset[150:300]
inx3 = trainset[300::]

perfK = np.zeros([len(K),3])

for ki in range(len(K)):
    k = K[ki]
    print k
                    
    # - LOOP OVER DIFFERENT COMBINATIONS OF inx1,inx2,inx3
    for t in range(3):
        if t == 0: t_train = np.concatenate((inx1, inx2)); t_val = inx3;
        elif t == 1: t_train = np.concatenate((inx2, inx3)); t_val = inx1;
        else: t_train = np.concatenate((inx1, inx3)); t_val = inx2;
    
        # - MEASURE THE MEAN CLASSIFICATION ACCURACY FOR ALL IMAGES IN THE VALIDATION PART    

        perfK[ki,t] = np.random.rand() #This is random!

# - PICK THE BEST K AS THE VALUE OF K THAT WORKS BEST ON AVERAGE FOR ALL POSSIBLE
mPerfK = mean(perfK,1)
Kbest  = K[np.argmax(mPerfK)]




# PART 3. SVM ON TOY DATA
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data,labels)

#*****
# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS
#*****
# Use Eq 3 (from the handout) for the prediction
# np.inner can help by computing the inner product between a vector and the weights
pred = ...

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

#*****
#Q6: THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA
#*****
C = 1
svm.LinearSVC( ... )
# OR similarly svc = svm.SVC(kernel='linear', ... )
pred = ...


# PART 4. SVM ON RING DATA
data, labels = week56.generate_ring_data()

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')

# Q7: USE LINEAR SVM AS BEFORE, VISUALIZE RESULTS and DRAW PREFERRED CLASSIFICATION LINE IN FIGURE

# Q8: (report only) 

# Q9: TRANSFORM DATA TO POLAR COORDINATES FIRST
rad = 
ang = 
# PLOT POLAR DATA

data2 = np.vstack((rad, ang))
data2 = data2.T

# Q10: USE THE LINEAR SVM AS BEFORE (BUT ON DATA 2)

# PLOT THE RESULTS IN ORIGINAL DATA

# PLOT POLAR DATA



# PART 5. LOAD BAG-OF-WORDS FOR THE OBJECT IMAGES AND RUN SVM CLASSIFIER FOR THE OBJECTS
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

K = 500
bow = np.zeros([len(files), K])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(K) + '/' + filnam2 + '/' + filnam + '.pkl')

#*****
# Q11: USE linear SVM, perform CROSS VALIDATION ON C = (.1,1,10,100), evaluate using MEAN CLASS ACCURACY
#*****
# Advice (from Piazza): use a broader range of C to observe real differences:
# For example C = [1e-5, 1e-2, 1, 1e2, 1e5]
svc = svm.LinearSVC (...)

# Here we do *multi-class* SVM which works slightly different.
# It is beyond this lecture / lab to discuss how the prediction is performed exactly*
# But, luckily it is implemented for us:
pred = svc.predict(bow[trainset])
# Since this is the prediction on the trainset, we should obtain a very good prediction! in general
# *Multi-class classification is a topic of the bonus, there you will need to think about how you could perform multi-class classification using a set of binary classifiers


# Q12: Visualize the best performing SVM, what are good classes, bad classes, examples of images etc

# Q13: Compare SVM with k-NN




