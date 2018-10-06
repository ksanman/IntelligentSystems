#!/usr/bin/python

####################################################
# module: mnist_digits_decision_tree.py
# description: decision trees for MNIST
# bugs to vladimir dot kulyukin at usu dot edu
####################################################

from sklearn import tree, metrics
import numpy as np
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

## ensure that the data have bee reshaped correctly.
for i in xrange(len(mnist_train_data)):
    assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                          mnist_train_data_dc[i])

for i in xrange(len(mnist_test_data)):
    assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                          mnist_test_data_dc[i])

for i in xrange(len(mnist_valid_data)):
    assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                          mnist_valid_data_dc[i])

# set the values of the target arrays.
mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
mnist_test_target_dc  = reshape_mnist_target2(mnist_test_data)
mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data)

from sklearn.metrics import confusion_matrix, classification_report

# this function tests a decision tree and computes a classification
# report and a confusion matrix.
def test_dtr(valid_data):
    clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 100))
    dtr = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    print 'Training completed...'
    valid_preds = dtr.predict(valid_data)
    print metrics.classification_report(mnist_valid_target_dc, valid_preds)
    #cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)
    #print cm1

if __name__ == '__main__':
    test_dtr(mnist_valid_data_dc)


