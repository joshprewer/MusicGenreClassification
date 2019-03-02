import scipy.stats as st
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import librosa
from rp_extract import rp_extract

fileName = 'SSD+RPFeaturesGTZAN.csv'

class feature_vector:
    def __init__(self, name):
        self.mean = f'{name}'
        self.variance = f'{name}_var'
        self.median= f'{name}_med'
        self.skewness = f'{name}_skew'
        self.kurtosis = f'{name}_kurt'
        self.min = f'{name}_min'
        self.max = f'{name}_max'

        self.function = None
        self.data = None

    def get_features(self):
        return f' {np.mean(self.data)} {np.std(self.data)} {np.median(self.data)} {st.skew(self.data)} {st.kurtosis(self.data)} {np.min(self.data)} {np.max(self.data)}'


ssd_features = []
for i in range(0, 24):
    ssd_features = np.append(ssd_features, feature_vector(f'ssd{i}'))

header = 'filename'

for feature in ssd_features:
    header += f' {feature.mean} {feature.variance} {feature.median} {feature.skewness} {feature.kurtosis} {feature.min} {feature.max}'

for i in range (0, 60):
    header += f' rh{i}'

header += ' label'
header = header.split()
    
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in sorted(os.listdir(f'./Datasets/GTZAN/audio/{g}')):
        songname = f'./Datasets/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'
        
        ssd_rp = rp_extract.rp_extract(wavedata=y, samplerate=sr, extract_ssd=True, extract_rh=True)
        ssd = ssd_rp.get('ssd')
        rp = ssd_rp.get('rh')

        for item in ssd:
            to_append += f' {item}'

        for item in rp:
            to_append += f' {item}'
        
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

