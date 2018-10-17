import numpy as np
import tensorflow
from tensorflow import reset_default_graph
import training_nets as tn
import net_tester as nt 

BEE_TRAIN_DIR = '/datasets/project1/data/bee_images/BEE2Set/bee_train'
NO_BEE_TRAIN_DIR = '/datasets/project1/data/bee_images/BEE2Set/no_bee_train'
BUZZ_TRAIN_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/train/bee_train'
CRICKET_TRAIN_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/train/cricket_train'
NOISE_TRAIN_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/train/noise_train'

BEE_TEST_DIR = '/datasets/project1/data/bee_images/BEE2Set/bee_test'
NO_BEE_TEST_DIR = '/datasets/project1/data/bee_images/BEE2Set/no_bee_test'
BUZZ_TEST_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/test/bee_test'
CRICKET_TEST_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/test/cricket_test'
NOISE_TEST_DIR = '/datasets/project1/data/bee_sounds/BUZZ2Set/test/noise_test'

def test_model(model, x, y):
    results = []
    prediction = np.array( model.predict(x))
    prediction = prediction.round(decimals=0).astype(int)
    print prediction
    print y
    for i in range(len(prediction)):
        if (prediction[i] == y[i]).all():
            results.append(True)
       # print 'p',prediction[i]
       # print 't',y[i]
       # results.append((prediction[i] == y[i]).all()
    res = float(len(results))/float(len(prediction))
    print res
    return res

#from time import time
#
#s = time()
#tn.train_image_ann()
#f = time()
#ann_t = f - s
#s = time()
#tn.train_image_cnn()
#f = time()
#cnn_t = f - s
#s = time()
#tn.train_audio_ann()
#f = time()
#audio_a_t = f - s
#s = time()
#tn.train_audio_cnn()
#f = time()
#audio_c_t = f - s

reset_default_graph()
image_ann = tn.load_image_ann()
res = nt.fit_image_ann(image_ann, '/datasets/project1/data/bee_images/BEE2Set/bee_train/img0/100_4_yb.png')
print res
raw_input("press any key to continue")

reset_default_graph()
image_cnn = tn.load_image_cnn()
reset_default_graph()
audio_ann = tn.load_audio_ann()
reset_default_graph()
audio_cnn = tn.load_audio_cnn()

bees = tn.collect_ANN_training_image_set(BEE_TEST_DIR)
no_bees = tn.collect_ANN_training_image_set(NO_BEE_TEST_DIR)
print len(bees)
print len(no_bees)
bee_y = [[1,0] for x in range(len(bees))]
no_bee_y = [[0,1] for x in range(len(no_bees))]
X = np.append(bees, no_bees,axis=0)
Y = np.append(bee_y, no_bee_y,axis=0)
 
print X.shape
ann_acc = test_model(image_ann, X, Y)

bees = tn.collect_CNN_training_image_set(BEE_TEST_DIR)
no_bees = tn.collect_CNN_training_image_set(NO_BEE_TEST_DIR)
bee_y = [[1,0] for x in range(len(bees))]
no_bee_y = [[0,1] for x in range(len(no_bees))]
X = np.append(bees, no_bees,axis=0)
Y = np.append(bee_y, no_bee_y,axis=0)
 
print X.shape
cnn_acc = test_model(image_cnn, X, Y)

buzz_d = tn.collect_audio_ann_training_data(BUZZ_TEST_DIR)
chirp_d = tn.collect_audio_ann_training_data(CRICKET_TEST_DIR)
noise_d = tn.collect_audio_ann_training_data(NOISE_TEST_DIR)
print len(buzz_d)
print len(chirp_d)
print len(noise_d)
buzz_y = [[1,0,0] for x in range(len(buzz_d))]
chirp_y = [[0,1,0] for x in range(len(chirp_d))]
noise_y = [[0,0,1] for x in range(len(noise_d))]
X = np.append(np.append(buzz_d,chirp_d, axis=0), noise_d, axis=0)
Y = np.append(np.append(buzz_y, chirp_y, axis=0),noise_y, axis=0)
    
 
print X.shape
audio_ann_acc = test_model(audio_ann, X, Y)

buzz_d = tn.collect_audio_cnn_training_data(BUZZ_TEST_DIR)
chirp_d = tn.collect_audio_cnn_training_data(CRICKET_TEST_DIR)
noise_d = tn.collect_audio_cnn_training_data(NOISE_TEST_DIR)
buzz_y = [[1,0,0] for x in range(len(buzz_d))]
chirp_y = [[0,1,0] for x in range(len(chirp_d))]
noise_y = [[0,0,1] for x in range(len(noise_d))]
X = np.append(np.append(buzz_d,chirp_d, axis=0), noise_d, axis=0)
Y = np.append(np.append(buzz_y, chirp_y, axis=0),noise_y, axis=0)
 

print X.shape
audio_cnn_acc = test_model(audio_cnn, X, Y)

print ann_acc
print cnn_acc
print audio_ann_acc
print audio_cnn_acc

f = open('stats.txt','a+')
#f.write('Image ANN training time: ')
#f.write(str(ann_t))
#f.write('\nImage CNN Training time: ')
#f.write(str(cnn_t))
#f.write('\nAudio ANN Training time: ')
#f.write(str(audio_a_t))
#f.write('\nAudio CNN training time: ')
#f.write(str(audio_c_t))
#f.write('\n')
f.write(str(ann_acc))
f.write('\n')
f.write(str(cnn_acc))
f.write('\n')
f.write(str(audio_ann_acc))
f.write('\n')
f.write(str(audio_cnn_acc))
f.write('\n')
f.close()

