import cv2
import numpy as np
from scipy.io import wavfile
from scipy.fftpack  import fft
import tflearn as tf
from tensorflow import reset_default_graph
from scipy.signal import spectrogram
import os
import fnmatch as fn

BEE_TRAIN_DIR = 'bee_images/BEE2Set/bee_train'
NO_BEE_TRAIN_DIR = 'bee_images/BEE2Set/no_bee_train'
BUZZ_TRAIN_DIR = 'bee_sounds/BUZZ2Set/train/bee_train'
CRICKET_TRAIN_DIR = 'bee_sounds/BUZZ2Set/train/cricket_train'
NOISE_TRAIN_DIR = 'bee_sounds/BUZZ2Set/train/noise_train'


def read_and_scale_CNN_image(image_path):
    img = (cv2.imread(image_path)/float(255))
    return img

def read_and_scale_ANN_image(image_path):
     img = cv2.imread(image_path)
     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     scaled_gray_image = gray_image/255
     reshaped = np.reshape(scaled_gray_image, (1024))
     return reshaped

def collect_CNN_training_image_set(set_dir):
    files = []
    for root, directories, filenames in os.walk(set_dir):
       for directory in directories:
           for file in os.listdir(set_dir + '/' + directory):
               if fn.fnmatch(file, '*.png'):
                   files.append(read_and_scale_CNN_image(set_dir + '/' + directory + '/' + file))
    return files


def collect_ANN_training_image_set(set_dir):
   files = []
   for root, directories, filenames in os.walk(set_dir):
       for directory in directories:
           for file in os.listdir(set_dir + '/' + directory):
               if fn.fnmatch(file, '*.png'):
                   files.append(read_and_scale_ANN_image(set_dir + '/' + directory + '/' + file))
   return files


def read_and_scale_audio(image_path):
    samplerate, audio = wavfile.read(image_path)
    if audio.size < 88245:
        new_size = 88244 - audio.size
        audio_addition = np.zeros(new_size)
        audio = np.append(audio, audio_addition)
    x_size = 16000
    img_size = 128
    window_length = 512
    window_shift = 121
    
    if len(audio) > x_size:
        audio = audio[:x_size]

    X = np.zeros(x_size).astype('float32')
    X[:len(audio)] += audio
    spec = np.zeros((img_size, img_size)).astype('float32')

    for i in range(img_size):
        start = i * window_shift
        end = start + window_length
        sig = np.abs(np.fft.rfft(X[start:end] * np.hanning(window_length)))
        spec[:,i] = (sig[1:img_size + 1])[::-1]
    
    spec = (spec-spec.min())/(spec.max()-spec.min())
    spec = np.log10((spec * 100 + 0.01))
    spec = (spec-spec.min())/(spec.max()-spec.min()) - 0.5
    return spec.reshape(16384)

def collect_audio_training_data(set_dir):
    files = []
    for root, directories, filenames in os.walk(set_dir):
       for file in filenames:
            files.append(read_and_scale_audio(set_dir + '/' + file))
    return np.array(files)

def train_image_ann():
    bees = collect_ANN_training_image_set(BEE_TRAIN_DIR)
    no_bees = collect_ANN_training_image_set(NO_BEE_TRAIN_DIR)
    bee_y = [[1,0] for x in range(len(bees))]
    no_bee_y = [[0,1] for x in range(len(no_bees))]
    X = np.concatenate([bees[:5000], no_bees[:5000]])
    Y = np.concatenate([bee_y[:5000], no_bee_y[:5000]])
    reset_default_graph()
    model = build_image_ann_model([1024,1024,2048,64], 0.0001)
    model.fit(X, Y, 500, validation_set=0.25, batch_size=1000, shuffle=True, show_metric=True)
    pred = model.predict(bees)
    pred2 = model.predict(no_bees)
    model.save('nets/image_ann.tf')

def train_image_cnn():
    bees = collect_CNN_training_image_set(BEE_TRAIN_DIR)
    no_bees = collect_CNN_training_image_set(NO_BEE_TRAIN_DIR)
    bee_y = [[1,0] for x in range(len(bees))]
    no_bee_y = [[0,1] for x in range(len(no_bees))]
    X = np.concatenate([bees[:5000], no_bees[:5000]])
    Y = np.concatenate([bee_y[:5000], no_bee_y[:5000]])
    reset_default_graph()
    model = build_image_cnn_model()
    model.fit(X, Y, 10, validation_set=0.25, batch_size=100, shuffle=True,show_metric=True)
    model.save('nets/image_cnn.tf')

def train_audio_ann():
    buzz_d = np.array(collect_audio_training_data(BUZZ_TRAIN_DIR))
    chirp_d = np.array(collect_audio_training_data(CRICKET_TRAIN_DIR))
    noise_d = np.array(collect_audio_training_data(NOISE_TRAIN_DIR))
    buzz_y = [[1,0,0] for x in range(len(buzz_d))]
    chirp_y = [[0,1,0] for x in range(len(chirp_d))]
    noise_y = [[0,0,1] for x in range(len(noise_d))]
    a1 = np.append(buzz_d, chirp_d, axis=0)
    X = np.append(a1, noise_d, axis=0)
    Y = np.append(np.append(buzz_y, chirp_y, axis=0),noise_y, axis=0)
    reset_default_graph()
    model = build_audio_ann_model()
    model.fit(X, Y, 10, validation_set=0.25, batch_size=100, shuffle=True, show_metric=True)
    #model.save('nets/audio_ann.tf')

def train_convnet_cnn():
    buzz_d = collect_audio_training_data(BUZZ_TRAIN_DIR)
    chirp_d = collect_audio_training_data(CRICKET_TRAIN_DIR)
    noise_d = collect_audio_training_data(NOISE_TRAIN_DIR)
    buzz_y = [[1,0,0] for x in range(len(buzz_d))]
    chirp_y = [[0,1,0] for x in range(len(chirp_d))]
    noise_y = [[0,0,1] for x in range(len(noise_d))]
    X = np.concatenate([buzz_d[:100], chirp_d[:100], noise_d[:100]])
    Y = np.concatenate([buzz_y[:100], chirp_y[:100],noise_y[:100]])
    reset_default_graph()
    model = build_audio_cnn_model()
    model.fit(X, Y, 20, validation_set=0.10, batch_size=10, shuffle=True)
    #model.save('nets/audio_cnn.tf')

#Builds a 2048x512x32x2 neural net model and trains it with a learning rate of 0.01
def build_image_ann_model(layers, lr):
    # Input is 32 X 32, black and white images   
    input_layer = tf.input_data(shape=[None, 1024])
    for l in layers:
        net = tf.fully_connected(input_layer, l, activation='tanh', regularizer='L2', weights_init='xavier')
        net = tf.dropout(net, 0.6)
    net = tf.fully_connected(net, 2, activation='softmax', weights_init='xavier')
    net = tf.regression(net, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')
    model = tf.DNN(net)
    return model

def load_image_ann():
    reset_default_graph()
    model = build_image_ann_model()
    model.load('nets/image_ann.tf')

#Builds a 64x32x16 convolutional neural net model and trains it with a learning rate of 0.01
def build_image_cnn_model():
    # takes a 32x32x3 image
    net = tf.input_data(shape=[None, 32,32,3], name='Input')
    #Convolution layers
    for l in [32,32,64]:
        net = tf.conv_2d(net, nb_filter=l, filter_size=(3,3), activation='relu')
        net = tf.max_pool_2d(net, (2,2))
        #net = tf.dropout(net, 0.5)
        #net = tf.local_response_normalization(net)
    net = tf.flatten(net)
    net = tf.fully_connected(net, 64, activation='relu')
    net = tf.fully_connected(net, 2, activation='softmax')
    net = tf.regression(net, learning_rate=0.0001)
    model = tf.DNN(net)
    return model

def load_image_cnn():
    reset_default_graph()
    model = build_image_cnn_model()
    model.load('nets/image_cnn.tf')

def build_audio_ann_model():
    net = tf.input_data(shape=[None, 16384], name='Input')
    for l in [128,128,64]:
        net = tf.fully_connected(net, l, activation='sigmoid')

    net = tf.fully_connected(net, 3, activation='softmax')
    net = tf.regression(net, learning_rate=0.001)
    model = tf.DNN(net) 
    return model

def load_audio_ann():
    reset_default_graph()
    model = build_audio_ann_model()
    model.load('nets/audio_ann.tf')

def build_audio_cnn_model():
    net = tf.input_data(shape=[None, 32,32,3], name='Input')
    for l in [32, 32, 10]:
        covn = tf.conv_2d(input, nb_filter=l, filter_size=5, activation='tanh')
        max_pool = tf.max_pool_2d(cov1, 2)
        lrn = tf.local_response_normalization(max_pool)

    net = tf.fully_connected(net, 3, activation='softmax')
    net = tf.regression(net, learning_rate=0.01)
    model = tf.DNN(net)
    return model

def load_audio_cnn():
    reset_default_graph()
    model = build_audio_cnn_model()
    model.load('nets/audio_cnn.tf')

#train_image_ann()
#train_image_cnn()
train_audio_ann()
#train_audio_cnn()