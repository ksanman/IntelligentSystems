#!/usr/bin/python

from __future__ import division, print_function

################################################
# module: mnist_convnet_load.py
# Kody Sanchez
# A01514541
# bugs to vladimir dot kulyukin at usu dot edu
################################################

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np
import pickle as cPickle
from scipy import stats

# change these paths accordingly
net_path_1 = 'my_mnist_net/hw05_my_net_01.tfl'
net_path_2 = 'my_mnist_net/hw05_my_net_02.tfl'
net_path_3 = 'my_mnist_net/hw05_my_net_03.tfl'
net_path_4 = 'my_mnist_net/hw05_my_net_04.tfl'
net_path_5 = 'my_mnist_net/hw05_my_net_05.tfl'

def load_mnist_convnet_1(path):
    ## your code here
    input = tflearn.input_data(shape=[None, 28, 28, 1])
    cov1 = tflearn.conv_2d(input, nb_filter=64, filter_size=5, activation='sigmoid', name='conv_layer_1')
    max_pool1 = tflearn.max_pool_2d(cov1, 2, name='pool1')
    lrn = tflearn.local_response_normalization(max_pool1)
    cov2 = tflearn.conv_2d(lrn, nb_filter=32, filter_size=5, activation='sigmoid', name='conv_layer_2')
    max_pool2 = tflearn.max_pool_2d(cov2, 2, name='pool2')
    fc1 = tflearn.fully_connected(max_pool2, 100, activation='sigmoid')
    fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
    model = tflearn.DNN(fc2)
<<<<<<< HEAD
    model.load(path,weights_only=True)
    return model
=======
    return model.load(net_path_1)
>>>>>>> 84fc937480dcf39407d9daf7e3d1b3e7748e187a

def load_mnist_convnet_2(path):
    ## your code here
    input = tflearn.input_data(shape=[None, 28, 28, 1])
    cov1 = tflearn.conv_2d(input, nb_filter=32, filter_size=5, activation='sigmoid', name='conv_layer_1')
    max_pool1 = tflearn.max_pool_2d(cov1, 2, name='pool1')
    lrn = tflearn.local_response_normalization(max_pool1)
    cov2 = tflearn.conv_2d(lrn, nb_filter=64, filter_size=5, activation='sigmoid', name='conv_layer_2')
    max_pool2 = tflearn.max_pool_2d(cov2, 2, name='pool2')
    fc1 = tflearn.fully_connected(max_pool2, 100, activation='sigmoid')
    fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
    model =  tflearn.DNN(fc2)
<<<<<<< HEAD
    model.load(path,weights_only=True)
    return model
=======
    return model.load(net_path_2)
>>>>>>> 84fc937480dcf39407d9daf7e3d1b3e7748e187a

def load_mnist_convnet_3(path):
    ## your code here
    input = tflearn.input_data(shape=[None, 28, 28, 1])
    cov1 = tflearn.conv_2d(input, nb_filter=32, filter_size=5, activation='relu', name='conv_layer_1')
    max_pool1 = tflearn.max_pool_2d(cov1, 2, name='pool1')
    lrn = tflearn.local_response_normalization(max_pool1)
    cov2 = tflearn.conv_2d(lrn, nb_filter=32, filter_size=5, activation='relu', name='conv_layer_2')
    max_pool2 = tflearn.max_pool_2d(cov2, 2, name='pool2')
    fc1 = tflearn.fully_connected(max_pool2, 100, activation='relu')
    fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
    model =  tflearn.DNN(fc2)
<<<<<<< HEAD
    model.load(path,weights_only=True)
    return model
=======
    return model.load(net_path_3)
>>>>>>> 84fc937480dcf39407d9daf7e3d1b3e7748e187a

def load_mnist_convnet_4(path):
    ## your code here
    input = tflearn.input_data(shape=[None, 28, 28, 1])
    cov1 = tflearn.conv_2d(input, nb_filter=128, filter_size=5, activation='sigmoid', name='conv_layer_1')
    max_pool1 = tflearn.max_pool_2d(cov1, 2, name='pool1')
    lrn = tflearn.local_response_normalization(max_pool1)
    cov2 = tflearn.conv_2d(lrn, nb_filter=64, filter_size=5, activation='sigmoid', name='conv_layer_2')
    max_pool2 = tflearn.max_pool_2d(cov2, 2, name='pool2')
    lrn2 = tflearn.local_response_normalization(max_pool2)
    cov3 = tflearn.conv_2d(lrn2, nb_filter=32, filter_size=5, activation='sigmoid', name='conv_layer_3')
    max_pool3 = tflearn.max_pool_2d(cov3, 2, name='pool3')
    fc1 = tflearn.fully_connected(max_pool3, 100, activation='sigmoid')
    fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
    model = tflearn.DNN(fc2)
<<<<<<< HEAD
    model.load(path,weights_only=True)
    return model
=======
    return model.load(net_path_4)
>>>>>>> 84fc937480dcf39407d9daf7e3d1b3e7748e187a

def load_mnist_convnet_5(path):
    ## your code here
    input = tflearn.input_data(shape=[None, 28, 28, 1])
    cov1 = tflearn.conv_2d(input, nb_filter=120, filter_size=10, activation='sigmoid', name='conv_layer_1')
    max_pool1 = tflearn.max_pool_2d(cov1, 2, name='pool1')
    lrn = tflearn.local_response_normalization(max_pool1)
    cov2 = tflearn.conv_2d(lrn, nb_filter=80, filter_size=10, activation='sigmoid', name='conv_layer_2')
    max_pool2 = tflearn.max_pool_2d(cov2, 2, name='pool2')
    lrn2 = tflearn.local_response_normalization(max_pool2)
    cov3 = tflearn.conv_2d(lrn2, nb_filter=40, filter_size=10, activation='sigmoid', name='conv_layer_3')
    max_pool3 = tflearn.max_pool_2d(cov3, 2, name='pool3')
    fc1 = tflearn.fully_connected(max_pool3, 30, activation='sigmoid')
    fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
    model =  tflearn.DNN(fc2)
<<<<<<< HEAD
    model.load(path,weights_only=True)
    return model
=======
    return model.load(net_path_5)
>>>>>>> 84fc937480dcf39407d9daf7e3d1b3e7748e187a

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = cPickle.load(fp)
    return obj

# load the validation data; change these directories accordingly
valid_x_path = 'my_mnist_net/valid_x.pck'
valid_y_path = 'my_mnist_net/valid_y.pck'
validX = load(valid_x_path)
validY = load(valid_y_path)

def test_convnet_model(convnet_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = convnet_model.predict(validX[i].reshape([-1, 28, 28, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return sum((np.array(results) == True))/len(results)


