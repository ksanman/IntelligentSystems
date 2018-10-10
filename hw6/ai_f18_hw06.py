#!/usr/bin/python

###############################
# module: ai_f18_hw06.py
# Kody Sanchez
# A01514541
###############################

import numpy as np
import pickle as cPickle
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
import random

from mnist_loader import load_data_wrapper

## load MNIST
mnist_train_data, mnist_test_data, mnist_valid_data = \
                  load_data_wrapper()

# define numpy arrays for MNIST data; dc stands for
# data conversion.
mnist_train_data_dc = np.zeros((50000, 784))
mnist_test_data_dc  = np.zeros((10000, 784))
mnist_valid_data_dc = np.zeros((10000, 784))

# define numpy arrays for MNIST targets
mnist_train_target_dc = None
mnist_test_target_dc  = None
mnist_valid_target_dc = None

# here is how we reshape mnist data for sklearn decision trees.
def reshape_mnist_d(mnist_data, mnist_data_dc):
    for i in xrange(len(mnist_data)):
        mnist_data_dc[i] = mnist_data[i][0].reshape((784,))

def reshape_mnist_data():
    global mnist_train_data
    global mnist_train_data_dc
    global mnist_test_data
    global mnist_test_data_dc
    global mnist_valid_data
    global mnist_valid_data_dc
    reshape_mnist_d(mnist_train_data, mnist_train_data_dc)
    reshape_mnist_d(mnist_test_data,  mnist_test_data_dc)
    reshape_mnist_d(mnist_valid_data, mnist_valid_data_dc)

# let's reshape the targets as well.
# we need 2 functions for this reshaping.
def reshape_mnist_target(mnist_data):
    return np.array([np.argmax(mnist_data[i][1])
                    for i in xrange(len(mnist_data))])

def reshape_mnist_target2(mnist_data):
    return np.array([mnist_data[i][1] for i in xrange(len(mnist_data))])

# actually reshape the data
reshape_mnist_data()

print 'mnist data reshaped...'

## ensure that the data have been reshaped correctly.
for i in xrange(len(mnist_train_data)):
    assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                          mnist_train_data_dc[i])

for i in xrange(len(mnist_test_data)):
    assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                          mnist_test_data_dc[i])

for i in xrange(len(mnist_valid_data)):
    assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                          mnist_valid_data_dc[i])

print 'mnist data verified...'

# set the values of the target arrays.
mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
mnist_test_target_dc  = reshape_mnist_target2(mnist_test_data)
mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data)

## get the data, the data items, and target
#digits_data = datasets.load_digits()
#data_items = digits_data.data
#data_target = digits_data.target

## Create your classifiers.
# an svm classifier with linear kernel
lin_svm = svm.SVC(kernel='linear')
# an svm classifier with rbf kernel
rbf_svm = svm.SVC(kernel='rbf')
# an svm classifier with several degree poly kernels
poly_svm_2 = svm.SVC(kernel='poly', degree=2)
poly_svm_3 = svm.SVC(kernel='poly', degree=3)
poly_svm_4 = svm.SVC(kernel='poly', degree=4)
poly_svm_5 = svm.SVC(kernel='poly', degree=5)

print 'svms defined...'

def train_and_persist(svm_model, train_d, train_t, fp):
    # your code here
    # 1. train the classifier
    print 'Training svm model'
    
    from time import time
    st = time.now()
    svm_model.fit(train_d, train_t)
    et = time.now()
    duration = et - st

    with open("stats.txt", "a+") as myfile:
        myfile.write(duration)

    with open(fp, 'wb') as output:  # Overwrites any existing file.
        cPickle.dump(svm_model, output)
    print 'svm model trained and persisted...'

def print_svm_report(model_path, test_data, test_target):
    # your code here
    # load the model_path
    with open(model_path, 'rb') as output:
        svm = cPickle.load(output)

    predictions = svm.predict(test_data)

    data = str(svm) + metrics.classification_report(test_target, predictions)

    with open("stats.txt", "a+") as myfile:
        myfile.write(data)
    print("Classification report for SVM kernel %s:\n%s\n"
          % (svm, metrics.classification_report(test_target, predictions)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target, predictions))