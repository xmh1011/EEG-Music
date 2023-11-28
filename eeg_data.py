import glob
import os
import pickle
import time
import mne
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.signal import welch
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_filenames_in_folder(folder_path):
    filenames = glob.glob(os.path.join(folder_path, '*'))
    filenames = [os.path.basename(filename) for filename in filenames]
    return filenames


folder_path = './data/'

# 获取文件名数组
files = get_filenames_in_folder(folder_path)

hvha_data = []
hvla_data = []
lvha_data = []
lvla_data = []

for i in range(len(files)):
    # 读取.dat文件
    trial = pd.read_csv(folder_path + files[i], delimiter='\t')
    if files[i].split('.')[0].split('-')[1] == 'hvha':
        hvha_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'hvla':
        hvla_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'lvha':
        lvha_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'lvla':
        lvla_data.append(trial)

hvha_data = np.array(hvha_data)
hvla_data = np.array(hvla_data)
lvha_data = np.array(lvha_data)
lvla_data = np.array(lvla_data)
print(hvha_data.shape)

eeg_channels = np.array(
    ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
     'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2'])

left = np.array(["Fp1", "AF3", "F7", "FC5", "T7"])
right = np.array(["Fp2", "AF4", "F8", "FC6", "T8"])
frontal = np.array(["F3", "FC1", "Fz", "F4", "FC2"])
parietal = np.array(["P3", "P7", "Pz", "P4", "P8"])
occipital = np.array(["O1", "Oz", "O2", "PO3", "PO4"])
central = np.array(["CP5", "CP1", "Cz", "C4", "C3", "CP6", "CP2"])


def bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def get_band_power(eeg_data, trial, channel, band):
    bd = (0, 0)

    if band == "theta":  # drownsiness, emotional connection, intuition, creativity
        bd = (4, 8)
    elif band == "alpha":  # reflection, relaxation
        bd = (8, 12)
    elif band == "beta":  # concentration, problem solving, memory
        bd = (12, 30)
    elif band == "gamma":  # cognition, perception, learning, multi-tasking
        bd = (30, 64)

    return bandpower(eeg_data[trial, channel], 128, bd)
