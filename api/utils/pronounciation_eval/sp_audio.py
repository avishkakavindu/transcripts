import numpy as np
import scipy as sp
from scipy.io.wavfile import read  
import matplotlib.pyplot as plt
from scipy import signal
import sklearn.metrics.pairwise as metrics

import os.path as ops

def val_path(fpath):
    
    assert ops.exists(fpath), '{} file does not exist.'.format(fpath)

def rm_bg_noise(array, freq):
    b, a = signal.butter(5, 1000/(freq/2), btype = 'highpass')

    filtered_signal = signal.lfilter(b, a, array)
    plt.plot(filtered_signal)  
    plt.title('Highpass Filter')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')
    #plt.show()

    c, d = signal.butter(5, 380/(freq/2), btype = 'lowpass')
    filtered_signal = signal.lfilter(c, d, filtered_signal)
    plt.plot(filtered_signal)
    plt.title('Lowpass Filter')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')
    #plt.show()

    return filtered_signal

def load_audio(fpath):

    (Frequency, array) = read(fpath)

    return array[:, 1] if len(array.shape) > 1 else array, Frequency

def plot_waveform(array1, array2):

    plt.subplot(1, 2, 1)
    plt.plot(array1)
    plt.title('Audio Clip #1')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(array2)
    plt.title('Audio Clip #2')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')

    plt.show()

def gen_spectrogram(arr):
    spec, _, _, im = plt.specgram(arr, Fs = 1, noverlap=200)

    return spec, im

def plot_spectagram(array1, array2):
    ax1 = plt.subplot(221)
    plt.plot(array1)
    ax2 = plt.subplot(222)
    plt.plot(array2)
    plt.subplot(223, sharex = ax1)
    _, im1 = gen_spectrogram(array1)
    plt.subplot(224, sharex = ax2)
    _, im2 = gen_spectrogram(array2)
    
    plt.show()

def diff(fpath, arr, freq, plot_wave = False, plot_spec = False):
    val_path(fpath)

    arr1, _ = load_audio(fpath)

    filtered_arr2 = rm_bg_noise(arr, freq)
    arr2 = filtered_arr2[:, 1] if len(filtered_arr2.shape) > 1 else filtered_arr2

    spec1, _ = gen_spectrogram(arr1)
    spec2, _ = gen_spectrogram(arr2)

    if spec1.shape[1] > spec2.shape[1]:
        spec1 = spec1[:,:spec2.shape[1]]
    elif spec1.shape[1] < spec2.shape[1]:
        spec2 = spec2[:,:spec1.shape[1]] 

    # Calculate difference between two audios
    simi = sp.linalg.norm(metrics.pairwise_distances(spec1, spec2, metric = "canberra"))

    if plot_wave:
        plot_waveform(arr1, arr2)

    if plot_spec:
        plot_spectagram(arr1, arr2)

    return simi


def diff_base(fpath, spath):
    val_path(fpath)
    val_path(spath)

    arr1, _ = load_audio(fpath)
    arr2, freq2 = load_audio(spath)

    filtered_arr2 = rm_bg_noise(arr2, freq2)
    arr2 = filtered_arr2[:, 1] if len(filtered_arr2.shape) > 1 else filtered_arr2

    spec1, _ = gen_spectrogram(arr1)
    spec2, _ = gen_spectrogram(arr2)

    if spec1.shape[1] > spec2.shape[1]:
        spec1 = spec1[:,:spec2.shape[1]]
    elif spec1.shape[1] < spec2.shape[1]:
        spec2 = spec2[:,:spec1.shape[1]] 

    # Calculate difference between two audios
    simi = sp.linalg.norm(metrics.pairwise_distances(spec1, spec2, metric = "canberra"))

    return simi