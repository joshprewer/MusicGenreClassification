import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
from scipy.stats import kurtosis, skew

fileName = 'SpectralFeaturesGTZAN.csv'

class feature_vector:
    def __init__(self, name, function):
        self.mean = f'{name}'
        self.variance = f'{name}_v'
        self.kurt = f'{name}_kurt'
        self.skew = f'{name}_skew'

        self.function = function
        self.data = []


features = [feature_vector('rmse', librosa.feature.rmse),
            feature_vector('spectral_centroid', librosa.feature.spectral_centroid), feature_vector('spectral_bandwidth', librosa.feature.spectral_bandwidth), 
            feature_vector('spectral_contrast', librosa.feature.spectral_contrast), feature_vector('spectral_flatness', librosa.feature.spectral_flatness),
            feature_vector('spectral_rolloff', librosa.feature.spectral_rolloff), feature_vector('zcr', librosa.feature.zero_crossing_rate)]


header = 'filename'
for feature in features:
    header += f' {feature.mean} {feature.variance} {feature.kurt} {feature.skew}'


header += ' label'
header = header.split()
    
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./Datasets/GTZAN/audio/{g}'):
        songname = f'./Datasets/GTZAN/audio/{g}/{filename}'
        print(songname)
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'
        for feature in features:
            feature.data = feature.function(y=y)[0]
            to_append += f' {np.mean(feature.data)} {np.std(feature.data)} {kurtosis(feature.data)} {skew(feature.data)}'    

        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

