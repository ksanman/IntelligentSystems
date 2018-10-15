import numpy as np
import os
from scipy.io import wavfile
from scipy.fftpack import fft


def read_and_scale_audio(image_path):
    samplerate, audio = wavfile.read(image_path)
    if audio.size < 88245:
        new_size = 88244 - audio.size
        audio_addition = np.zeros(new_size)
        audio = np.append(audio, audio_addition)

    print('audio: ' + str(audio.shape))
    return np.array(audio)
    #scale the audio
    #audio = audio/float(np.max(audio))
    #scale using a fourier transform to get the spectrogram of the audio.
    #audio_fft = fft(audio)
    #return audio_fft

def collect_audio_training_data(set_dir):
    files = []
    for root, directories, filenames in os.walk(set_dir):
       for file in filenames:
            files.append(read_and_scale_audio(set_dir + '/' + file))
    return np.array(files)


dir1 = 'D:/Xaber/Documents/School/Fall 2018/CS5600 - Intelligent System/Homework/Project 1/Project 1/bee_sounds/BUZZ2Set/train/cricket_train'
dir2 = 'D:/Xaber/Documents/School/Fall 2018/CS5600 - Intelligent System/Homework/Project 1/Project 1/bee_sounds/BUZZ2Set/train/bee_train'
file1 = 'D:/Xaber/Documents/School/Fall 2018/CS5600 - Intelligent System/Homework/Project 1/Project 1/bee_sounds/BUZZ2Set/train/cricket_train/cricket1_192_168_4_6-2017-09-02_22-00-01.wav'
file2 = 'D:/Xaber/Documents/School/Fall 2018/CS5600 - Intelligent System/Homework/Project 1/Project 1/bee_sounds/BUZZ2Set/train/bee_train/192_168_4_6-2017-08-09_14-15-01_0.wav'

#cricketFiles = collect_audio_training_data(dir1)
buzzFiles = collect_audio_training_data(dir2)
print(cricketFiles.shape)
print(buzzFiles.shape)

