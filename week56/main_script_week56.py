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
import sklearn 
from sklearn import svm, datasets
import week56

# These values can be changed to your own liking
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

####
# Get object data
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()
####

###
# Load Bag-Of-Words
###
C = 100
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

###############################################################################
# Q1: IMPLEMENT HERE kNN CLASSIFIER. 
###############################################################################

# Normalize Bag-Of-Words
for i in range(len(files)):
    bow[i] = tools.normalizeL1(bow[i])

# k-NN Classifier
dist = np.zeros([len(testset),len(trainset)])
for j in range(len(testset)):
    for k in range(len(trainset)):
        b_j = bow[testset[j]]
        b_k = bow[trainset[k]]
        dist[j,k] = sum(np.minimum(b_j,b_k))

# Specify a maximum rank here. Default is 9.
K = 9

# Change image value here; (366 = goal), (150 = bicycle), (84 = beach), (450 = mountain).
query_id = int(150)

# Set up the ranking of images
qinx     = argmax(testset == query_id)
ranking = np.argsort(dist[argmax(testset == query_id), :])
ranking = ranking[::-1]
nearest_labels = labels[trainset[ranking[0 : K]]]

# Draw the results
figure
subplot(2, 6, 1)
imshow(Image.open(files[query_id]))
title('Query')
axis('off')

for cnt in range(K):
    subplot(2, 6, cnt+2)
    imshow(Image.open(files[trainset[ranking[cnt]]]))
    title(unique_labels[nearest_labels[cnt]-1])
    axis('off')

###############################################################################
# Q2: USE DIFFERENT STRATEGY
###############################################################################

###############################################################################
# Q3: For K = 9, COMPUTE THE CLASS ACCURACY FOR THE TESTSET
###############################################################################

# Predict labels for each image in the testset
K = 9 # For question 4, change the values here (1, 3, 5, 7, 9, 15)
pred = np.zeros(len(testset))
for i in range(len(testset)):
    ranking = np.argsort(dist[i, :])
    ranking = ranking[::-1]
    i_nearest_labels = labels[trainset[ranking[0 : K]]]
    i_label_count = np.bincount(i_nearest_labels)
    pred[i] = argmax(i_label_count)

# Compare the prediction with the labels of each class
labelstest = labels[testset]
classAcc = np.zeros(len(unique_labels))
allClass = np.zeros(len(unique_labels))
for c in range(len(testset)):
    if pred[c] == labelstest[c]:
        classAcc[labelstest[c] - 1] += 1
    allClass[labelstest[c] - 1] += 1

# Calculate the accuracy of each class
for c in range(len(unique_labels)):        
    classAcc[c] = classAcc[c] / (allClass[c] + 10e-10)
print "Accuracies per class:", classAcc

# Calculate the total average accuracy
print "Mean accuracy:", sum(classAcc) / 10

###############################################################################
# Q4: DO CROSS VALIDATION TO DEFINE PARAMETER K
###############################################################################

# We're going to set K as an array and calculate it all in one go
K = [1, 3, 5, 7, 9, 15]

# Create a duplicate of trainset var and then shuffle it
crossTrainset = trainset.copy()
random.shuffle(crossTrainset)

# Split them up
cross = [crossTrainset[0:166], crossTrainset[166:333], crossTrainset[333:500]]
possibleCombinations = [[0,1,2],[0,2,1],[1,2,0]]

# Store accuracies per class based on K
perfK = np.zeros([len(K), 3])
Kscore = np.zeros([len(K),3])

for ky in range(len(K)):
    k = K[ky]

    # Perform validation
    for kx in range(len(possibleCombinations)):
        randomList = possibleCombinations[kx]
        trainpart = np.concatenate([cross[randomList[0]],cross[randomList[1]]])
        validationpart = cross[randomList[2]]
 
        # Recalculate the distances
        dist_norm = np.zeros([len(validationpart), len(trainpart)])
        for y in range(len(validationpart)):
            for x in range(len(trainpart)):
                dist_norm[y,x] = sum(np.minimum(bow[validationpart[y]],bow[trainpart[x]]))

        # Predict the label for each image
        predicted_label = np.zeros(len(validationpart)).astype(int);
        for y in range(len(validationpart)):
            ranking = np.argsort(dist_norm[y,:])
            ranking = ranking[::-1]
            topK = labels[trainpart[ranking[0:k]]]
            counts = np.bincount(topK)
            predicted_label[y] = np.argmax(counts)

        # Determine the accuracy
        classAcc = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            subset = predicted_label[labels[validationpart] == c+1]
            tp = sum(subset == c+1)
            fn = sum(subset != c+1)
            classAcc[c] = tp / (tp + fn + 0.0)

        # Calculate The mean accuracy
        Kscore[ky, kx] = np.mean(classAcc)

        # Print label for each accuracy from left to right
        print "Airplane, Beach, Bicycle, Boat, Car, Flower, Goal, Mountain, Temple, Train", "\n", classAcc

# Set a maximum amount of results. The default is 6.
kmeanScore = np.zeros([6,1])
for y in range(Kscore.shape[0]):
    row = Kscore[y,:]
    meanScore = np.mean(row)
    kmeanScore[y] = meanScore

# Print the results
print "Accuracies per class:", '\n', Kscore
print "Mean accuracy:", '\n', kmeanScore
print "The most optimal K:", '\n', K[np.argmax(kmeanScore)]

###############################################################################
# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS
###############################################################################

# Used for this question
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data,labels)

# Start with 0.
class_i = 0
w = svm_w[class_i]
b = svm_b[class_i][0]

# Predict the classification
pred = sign(np.inner(w, data) + b)

# Match the classifier based on the amount of matches. The highest amount will be printed
match = sum(pred ==labels)
print match

# Calculate the accuracy by dividing the highest matches with the length of the label var
acc = match / double(len(labels))
print acc

# Draw the SVM
figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# Create the classifier and draw it.
X = np.array([min(data[:,0]),max(data[:,0])]);
Y = ((-svm_w[class_i][0]*X-svm_b[class_i][0][0])/double(svm_w[class_i][1]))
plt.plot(X,Y)

###############################################################################
# Q6: USE HERE SVC function from sklearn to run a linear svm
###############################################################################

# THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data,labels)

# SVM regularization parameter. The default is 1.
C = 1

# Use the SVM function to predict the label for the data
svc = svm.SVC(kernel = 'linear')
svm_line_train = svc.fit(data, labels)
pred = svc.predict(data)

# Draw the figure
figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# Get The seperating hyperplane
w = svc.coef_[0]
# Calculate the coefficiency and intercept
a = -w[0] / w[1]
xx = np.linspace(0, 2)
yy = a * xx - (svc.intercept_[0]) / w[1]
# Plot the line
plt.plot(xx, yy, 'k-')

###############################################################################
# Q7: USE LINEAR SVM AS BEFORE, VISUALIZE RESULTS and DRAW PREFERRED CLASSIFICATION LINE IN FIGURE
###############################################################################

# Load and generate the SVM ring data
data, labels = week56.generate_ring_data()

# Draw the SVM ring data
figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')

# Use the SVM function to predict the label for the data
svc = svm.SVC(kernel = 'linear')
svm_line_train = svc.fit(data, labels)
pred = svc.predict(data)

# Draw the figure
figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# We use the same function as before to draw the preferred classification line
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 2)
yy = a * xx - (svc.intercept_[0]) / w[1]

# Plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')

# Return an array filled with zeros
accuracy = np.zeros(2)

# Determine the accuracy for the green labels
green_class = labels == -1
print green_class
green_class_labels = labels[green_class]
print green_class_labels
green_class_pred = pred[green_class]
true_pos_green = sum(green_class_labels == green_class_pred)
accuracy[0] = true_pos_green / double(len(green_class_labels))

# Determine the accuracy for the red labels
red_class = labels == 1
red_class_labels = labels[red_class]
red_class_pred = pred[red_class]
true_pos_red = sum(red_class_labels == red_class_pred)
accuracy[1] = true_pos_red / double(len(red_class_labels))

# Print the accuracy of the green and red labels
print "The accuracy for green and red labels:", accuracy

print "The Amount of green labels:", len(green_class_labels)
print "The Amount of red labels:", len(red_class_labels)

###############################################################################
# Q8: (report only) 
###############################################################################

###############################################################################
# Q9: TRANSFORM DATA TO POLAR COORDINATES FIRST
###############################################################################

#  SVM regularization parameter
C = 1.0

# Generate the ring data
data, labels = week56.generate_ring_data()

# Use the radius and angle formulas
rad = sqrt((data[:,0]**2+data[:,1]**2))
ang = np.arctan2(data[:,1], data[:,0])

# Stacks the arrays sequencially
data2 = np.vstack((rad, ang))
data2 = data2.T

# Draw the figure
figure()
plt.scatter(data2[labels==1, 0], data2[labels==1, 1], facecolor='r')
plt.scatter(data2[labels==-1, 0], data2[labels==-1, 1], facecolor='g')

###############################################################################
# Q10: USE THE LINEAR SVM AS BEFORE (BUT ON DATA 2)
###############################################################################

# Plot the polar data
data2 = np.vstack((rad, ang))
data2 = data2.T

# Use the SVM function to predict the labels on data 2
svc = svm.SVC(kernel = 'linear')
svc.fit(data2, labels)
pred = svc.predict(data2)

# draw the results
plt.plot(data2[pred==1, 0], data2[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data2[pred==-1, 0], data2[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# Return an array filled with zeros
accuracy = np.zeros(2)

# Determine the accuracy for the green labels
green_class = labels == -1
print green_class
green_class_labels = labels[green_class]
print green_class_labels
green_class_pred = pred[green_class]
true_pos_green = sum(green_class_labels == green_class_pred)
accuracy[0] = true_pos_green / double(len(green_class_labels))

# Determine the accuracy for the red labels
red_class = labels == 1
red_class_labels = labels[red_class]
red_class_pred = pred[red_class]
true_pos_red = sum(red_class_labels == red_class_pred)
accuracy[1] = true_pos_red / double(len(red_class_labels))

# Print the accuracy for the green[0] and red[1] labels
print "The Accuracy for green and red labels:", accuracy

###############################################################################
# Q11: USE linear SVM, perform CROSS VALIDATION ON C = (.1,1,10,100), evaluate using MEAN CLASS ACCURACY
###############################################################################

# PART 5. LOAD BAG-OF-WORDS FOR THE OBJECT IMAGES AND RUN SVM CLASSIFIER FOR THE OBJECTS
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

# Codebook size. Changes these to 10, 100, 500, 1000, or 4000
J = 4000

# Load the Bag-Of-Words
bow = np.zeros([len(files), J])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(J) + '/' + filnam2 + '/' + filnam + '.pkl')
    bow[cnt] = tools.normalizeL1(bow[cnt], 0) # Don't forget to normalize it

# Return a array with zeros for the shape of the Bag-Of-Words
norm_bow = np.zeros(bow.shape)

for y in range(len(files)):
    norm_row = bow[y,:]/sum(bow[y,:])
    # of: row /= sum(row)
    bow[y,:] = norm_row

# Insert .1, 1, 10, or 100 here
C = [1e-5, 1e-2, 1, 1e2, 1e5]

# Create a duplicate of trainset var and then shuffle it
crossTrainset = trainset.copy()
random.shuffle(crossTrainset)

# Split them up
cross = [crossTrainset[0:166], crossTrainset[166:333], crossTrainset[333:500]]
possibleCombinations = [[0,1,2],[0,2,1],[1,2,0]]

# Store accuracies per class based on C
perfC = np.zeros([len(C), 3])
Cscore = np.zeros([len(C),3])

for cy in range(len(C)):
    c = C[cy]

    # Perform Validation
    for cx in range(len(possibleCombinations)):
        randomList = possibleCombinations[cx]
        trainpart = np.concatenate([cross[randomList[0]],cross[randomList[1]]])
        validationpart = cross[randomList[2]]

        # Use the SVM function to predict the labels of the Bag-Of-Words
        svc = svm.SVC(kernel = 'linear', C=c)
        svc.fit(bow[trainpart,:], labels[trainpart])
        pred = svc.predict(bow[validationpart,:])

        # Determine the accuracy
        classAcc = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            subset = pred[labels[validationpart] == c+1]
            tp = sum(subset == c+1)
            fn = sum(subset != c+1)
            classAcc[c] = tp / (tp + fn + 0.0)

        # Calculate the mean accuracy
        Cscore[cy, cx] = np.mean(classAcc)

        # Print label for each accuracy from left to right
        print "Airplane, Beach, Bicycle, Boat, Car, Flower, Goal, Mountain, Temple, Train", "\n", classAcc

# Set a maximum amount of results. The default is 5.
cmeanScore = np.zeros([5,1])
for y in range(Cscore.shape[0]):
    row = Cscore[y,:]
    meanScore = np.mean(row)
    cmeanScore[y] = meanScore

# Print the results
print "Accuracies per class:",  '\n', Cscore
print "Mean accuracy:", '\n', cmeanScore
print "The most optimal C:", '\n', C[np.argmax(cmeanScore)]

###############################################################################
# Q12: Visualize the best performing SVM, what are good classes, bad classes, examples of images etc 
###############################################################################

# This is the same code as in the previous questions
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

# Codebook size. Changes these to 10, 100, 500, 1000, or 4000
J = 500
bow = np.zeros([len(files), J])
cnt = -1
for impath in files:
    cnt = cnt + 1
    #print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :]  = week56.load_bow('../../data/bow_objects/codebook_' + str(J) + '/' + filnam2 + '/' + filnam + '.pkl')
    bow[cnt] = tools.normalizeL1(bow[cnt], 0)

C = [100000.0]
crossTrainset = trainset.copy()
random.shuffle(crossTrainset)

cross = [crossTrainset[0:166], crossTrainset[166:333], crossTrainset[333:500]]
perfC = np.zeros([len(C), 3])
possibleCombinations = [[0,1,2],[0,2,1],[1,2,0]]
Cscore = np.zeros([len(C),3])

for cy in range(len(C)):
    c = C[cy]

    for cx in range(len(possibleCombinations)):
        randomList = possibleCombinations[cx]
        trainpart = np.concatenate([cross[randomList[0]],cross[randomList[1]]])
        validationpart = cross[randomList[2]]
            
        svc = svm.SVC(kernel = 'linear', C=c)
        svc.fit(bow[trainpart, :] , labels[trainpart])
        pred = svc.predict(bow[validationpart, :])

        classSvm = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            subset = pred[labels[validationpart] == c+1]
            tp = sum(subset == c+1)
            fn = sum(subset != c+1)
            classSvm[c] = tp / (tp + fn + 0.0)

print "SVM Accuracy per class:", "\n", "Airplane, Beach, Bicycle, Boat, Car, Flower, Goal, Mountain, Temple, Train", "\n", classSvm
print 'SVM Mean accuracy:', "\n", np.mean(classSvm)

###############################################################################
# Q13: Compare SVM with k-NN
###############################################################################

# This is the same code used in the previous questions
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

# Codebook size. Changes these to 10, 100, 500, 1000, or 4000
J = 1000
bow = np.zeros([len(files), J])
cnt = -1
for impath in files:
    cnt = cnt + 1
    #print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(J) + '/' + filnam2 + '/' + filnam + '.pkl')
    bow[cnt] = tools.normalizeL1(bow[cnt], 0)

# Change K value here
K = [100000.0]
crossTrainset = trainset.copy()
random.shuffle(crossTrainset)

cross = [crossTrainset[0:166], crossTrainset[166:333], crossTrainset[333:500]]
perfK = np.zeros([len(K), 3])
possibleCombinations = [[0,1,2],[0,2,1],[1,2,0]]
Kscore = np.zeros([len(K),3])

for ky in range(len(K)):
    k = K[ky]

    for kx in range(len(possibleCombinations)):
        randomList = possibleCombinations[kx]
        trainpart = np.concatenate([cross[randomList[0]],cross[randomList[1]]])
        validationpart = cross[randomList[2]]

        dist_norm = np.zeros([len(validationpart), len(trainpart)])
        for y in range(len(validationpart)):
            for x in range(len(trainpart)):
                dist_norm[y,x] = sum(np.minimum(bow[validationpart[y]],bow[trainpart[x]]))
        
        predicted_label = np.zeros(len(validationpart)).astype(int);
        for y in range(len(validationpart)):
            ranking = np.argsort(dist_norm[y,:])
            ranking = ranking[::-1]
            topK = labels[trainpart[ranking[0:9]]]
            counts = np.bincount(topK)
            predicted_label[y] = np.argmax(counts)
            
        svc = svm.SVC(kernel = 'linear', C=k)
        svc.fit(bow[trainpart, :] , labels[trainpart])
        pred = svc.predict(bow[validationpart, :])

        classSvm = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            subset = pred[labels[validationpart] == c+1]
            tp = sum(subset == c+1)
            fn = sum(subset != c+1)
            classSvm[c] = tp / (tp + fn + 0.0)
            
        classKnn = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            subset = predicted_label[labels[validationpart] == c+1]
            tp = sum(subset == c+1)
            fn = sum(subset != c+1)
            classKnn[c] = tp / (tp + fn + 0.0)

print "SVM Accuracy per class:", "\n", "Airplane, Beach, Bicycle, Boat, Car, Flower, Goal, Mountain, Temple, Train", "\n", classSvm
print 'SVM Mean accuracy:', "\n", np.mean(classSvm), "\n"

print "k-NN Accuracy per class:", "\n", "Airplane, Beach, Bicycle, Boat, Car, Flower, Goal, Mountain, Temple, Train", "\n", classKnn
print 'k-NN Mean accuracy:', "\n", np.mean(classKnn)
