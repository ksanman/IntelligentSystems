import mnist_digits_decision_tree as dt
import mnist_digits_random_forest as rf
import time

#data = []
# tree_time = {}
# for i in range(10):
#     start = time.time()
#     dt.test_dtr(dt.mnist_valid_data_dc)
#     end = time.time()
#     tree_time['Decision Tree ' + str(i)] = end - start
#data.append(tree_time)
# forest_time = {}
# for i in range(10):
#     start = time.time()
#     rf.test_rf(10)
#     end = time.time()
#     forest_time['Forest ' + str(i)] = end - start
#data.append(forest_time)
# print tree_time, forest_time

start = time.time()
execfile("mnist_convnet_1.py")
execfile("mnist_convnet_2.py")
execfile("mnist_convnet_3.py")
execfile("mnist_convnet_4.py")
execfile("mnist_convnet_5.py")
en = time.time()
data.append({'net training time':en - start})
print en - start

import test_mnist_convnets as ts
net1 = ts.load_mnist_convnet_1(ts.net_path_1)
net2 = ts.load_mnist_convnet_2(ts.net_path_2)
net3 = ts.load_mnist_convnet_3(ts.net_path_3)
net4 = ts.load_mnist_convnet_4(ts.net_path_4)
net5 = ts.load_mnist_convnet_5(ts.net_path_5)
ac1 = ts.test_convnet_model(net1)
ac2 = ts.test_convnet_model(net2)
ac3 = ts.test_convnet_model(net3)
ac4= ts.test_convnet_model(net4)
ac5 = ts.test_convnet_model(net5)

data.append({'net1 acc: ' : ac1, 'net2 acc: ' : ac2, 'net3 acc: ':ac3, 'net4 acc:': ac4, 'net5 acc:':ac5})