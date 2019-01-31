import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os

fileName = 'SpectralFeaturesGTZAN.csv'

class feature_vector:
    def __init__(self, name, function):
        self.mean = f'{name}'
        self.variance = f'{name}_v'
        self.diff_mean = f'{name}_diff'
        self.diff_variance = f'{name}_diff_v'

        self.function = function
        self.data = None


features = [feature_vector('chroma_stft', librosa.feature.chroma_stft), feature_vector('rmse', librosa.feature.rmse),
            feature_vector('spectral_centroid', librosa.feature.spectral_centroid), feature_vector('spectral_bandwidth', librosa.feature.spectral_bandwidth), 
            feature_vector('spectral_contrast', librosa.feature.spectral_contrast), feature_vector('spectral_flatness', librosa.feature.spectral_flatness),
            feature_vector('spectral_rolloff', librosa.feature.spectral_rolloff), feature_vector('zcr', librosa.feature.zero_crossing_rate)]

mfcc_features = []
for i in range(1, 20):
    mfcc_features = np.append(mfcc_features, feature_vector(f'mfcc{i}', librosa.feature.mfcc))


header = 'filename'
for feature in features:
    header += f' {feature.mean} {feature.variance} {feature.diff_mean} {feature.diff_variance}'

for feature in mfcc_features:
    header += f' {feature.mean} {feature.variance} {feature.diff_mean} {feature.diff_variance}'

header += ' label'
header = header.split()

file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./MIR/GTZAN/audio/{g}'):
        songname = f'./MIR/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'
        for feature in features:
            feature.data = feature.function(y=y)
            to_append += f' {np.mean(feature.data)} {np.std(feature.data)} {np.mean(np.diff(feature.data))} {np.mean(np.diff(feature.data))}'    
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=20)
        for e in mfccs:
            to_append += f' {np.mean(e)} {np.std(e)} {np.mean(np.diff(e))} {np.std(np.diff(e))}'
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

