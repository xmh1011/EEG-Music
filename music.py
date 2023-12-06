import glob
import os
import numpy as np
import librosa
from pydub import AudioSegment


def get_filenames_in_folder(folder_path):
    filenames = glob.glob(os.path.join(folder_path, '*'))
    filenames = [os.path.basename(filename) for filename in filenames]
    return filenames


def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format='wav')


# 使用梅尔倒谱系数提取音频特征
def extract_audio_features(wav_file, sample_rate, num_mfcc):
    audio, sr = librosa.load(wav_file, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc) # 提取mfcc特征
    return mfcc.T


def get_music_features(file_path):
    files = os.listdir(file_path)
    audio_features = []

    for file in files:
        if file.endswith('.mp3'):
            mp3_file = os.path.join(file_path, file)
            wav_file = os.path.join(file_path, os.path.splitext(file)[0] + '.wav')
            convert_mp3_to_wav(mp3_file, wav_file)
            audio_feature = extract_audio_features(wav_file, sample_rate, frequency)
            audio_features.append(audio_feature)

    feature_1 = np.array(audio_features[0])
    feature_2 = np.array(audio_features[1])
    feature_3 = np.array(audio_features[2])
    feature_4 = np.array(audio_features[3])
    feature_5 = np.array(audio_features[4])

    features = np.concatenate((feature_1, feature_2, feature_3, feature_4, feature_5), axis=0)

    return features


lvla_file_path = r'./音频剪辑/抑郁'  # 抑郁
lvha_file_path = r'./音频剪辑/烦躁'  # 烦躁
hvla_file_path = r'./音频剪辑/放松'  # 放松
hvha_file_path = r'./音频剪辑/兴奋'  # 兴奋
sample_rate = 44100  # 采样率，与WAV文件的采样率相匹配
frequency = 1000  # 欲提取的频率

lvla_audio_features = get_music_features(lvla_file_path)
lvha_audio_features = get_music_features(lvha_file_path)
hvla_audio_features = get_music_features(hvla_file_path)
hvha_audio_features = get_music_features(hvha_file_path)
print("lvla:")
print(lvla_audio_features.shape)
print("lvha:")
print(lvha_audio_features.shape)
print("hvla:")
print(hvla_audio_features.shape)
print("hvha:")
print(hvha_audio_features.shape)