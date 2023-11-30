import numpy as np
import librosa
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format='wav')


def extract_audio_features(wav_file, sample_rate, frequency):
    audio, sr = librosa.load(wav_file, sr=sample_rate)
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    frequency_bin = int(frequency * stft.shape[0] / sr)
    frequency_samples = magnitude[frequency_bin, :]
    return frequency_samples


mp3_file = '音乐.mp3'  # 输入的MP3文件路径
wav_file = 'output.wav'  # 输出的WAV文件路径
sample_rate = 44100  # 采样率，与WAV文件的采样率相匹配
frequency = 1000  # 欲提取的频率

convert_mp3_to_wav(mp3_file, wav_file)
audio_features = extract_audio_features(wav_file, sample_rate, frequency)

audio_features = np.array(audio_features)
