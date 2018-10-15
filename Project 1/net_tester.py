import training_nets as tn

def fit_image_ann(ann, image_path):
    img = tn.load_image_ann(image_path)
    return ann.predict(img)

def fit_image_convnet(convnet, image_path):
    img = tn.load_image_cnn(image_path)
    return convnet.predict(img)

def fit_audio_ann(ann, audio_path):
    ad = tn.build_audio_ann_model(audio_path)
    return ann.predict(ad)

def fit_audio_convnet(convnet, audio_path):
    ad = tn.build_audio_cnn_model(audio_path)
    return convnet.predict(ad)