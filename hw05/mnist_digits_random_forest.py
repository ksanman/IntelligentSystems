#!/usr/bin/python

####################################################
# module: digits_decision_tree.py
# description: A random forest for MNIST datasets
# bugs to vladimir dot kulyukin at usu dot edu
####################################################

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from mnist_loader import load_data_wrapper
from sklearn import tree, metrics


# load MNIST
mnist_train_data, mnist_test_data, mnist_valid_data = \
                  load_data_wrapper()

# define reshaped data and targets for SKLEARN
mnist_train_data_dc = np.zeros((50000, 784))
mnist_test_data_dc  = np.zeros((10000, 784))
mnist_valid_data_dc = np.zeros((10000, 784))

mnist_train_target_dc = None
mnist_test_target_dc  = None
mnist_valid_target_dc = None

# functions that reshape the data from MNIST format
# to SKLearn format
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

def reshape_mnist_target(mnist_data):
    return np.array([np.argmax(mnist_data[i][1])
                    for i in xrange(len(mnist_data))])

def reshape_mnist_target2(mnist_data):
    return np.array([mnist_data[i][1] for i in xrange(len(mnist_data))])
                     
reshape_mnist_data()

# ensure that the data have been reshaped correctly.
for i in xrange(len(mnist_train_data)):
    assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                          mnist_train_data_dc[i])

for i in xrange(len(mnist_test_data)):
    assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                          mnist_test_data_dc[i])

for i in xrange(len(mnist_valid_data)):
    assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                          mnist_valid_data_dc[i])

# assign the values of the targets.
mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
mnist_test_target_dc  = reshape_mnist_target2(mnist_test_data)
mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data)


from sklearn.metrics import confusion_matrix, classification_report
#from matplotlib import pylab

# train and test a random forest with num_trees and print
# a classification report.
def test_rf(num_trees):
    clf = RandomForestClassifier(n_estimators=num_trees,
                                 random_state=random.randint(0, 1000))
    rf = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    print 'Training completed...'

    valid_preds = rf.predict(mnist_valid_data_dc)
    print metrics.classification_report(mnist_valid_target_dc, valid_preds)
    #cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)
    #print cm1

    #test_preds = rf.predict(mnist_test_data_dc)
    #print metrics.classification_report(mnist_test_target_dc, test_preds)
    #cm2 = confusion_matrix(mnist_test_target_dc, test_preds)
    #print cm2




