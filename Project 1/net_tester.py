import training_nets as tn
import numpy as np

def fit_image_ann(ann, image_path):
    img = [tn.read_and_scale_ANN_image(image_path)]
    prediction = np.array(ann.predict(img))
    prediction = prediction.round(decimals=0).astype(int)
    return prediction

def fit_image_convnet(convnet, image_path):
    img = [tn.read_and_scale_CNN_image(image_path)]
    prediction = np.array(convet.predict(img))
    prediction = prediction.round(decimals=0).astype(int)
    return prediction

def fit_audio_ann(ann, audio_path):
    ad = [tn.read_and_scale_ann_audio(audio_path)]
    prediction = np.array(ann.predict(ad))
    prediction = prediction.round(decimals=0).astype(int)
    return prediction

def fit_audio_convnet(convnet, audio_path):
    ad = [tn.read_and_scale_cnn_audio(audio_path)]
    prediction = np.array(convet.predict(ad))
    prediction = prediction.round(decimals=0).astype(int)
    return prediction
