import scipy.stats as st
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import librosa
from rp_extract import rp_extract

fileName = 'FeaturesGTZAN.csv'

class ssd_feature_vector:
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
    ssd_features = np.append(ssd_features, ssd_feature_vector(f'ssd{i}'))

header = 'filename'

for feature in ssd_features:
    header += f' {feature.mean} {feature.variance} {feature.median} {feature.skewness} {feature.kurtosis} {feature.min} {feature.max}'

for i in range (0, 24):
    header += f' rp{i}'
    header += f' rp{i}_std'

# header += ' label'
# header = header.split()
    
# file = open(fileName, 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in sorted(os.listdir(f'./Datasets/GTZAN/audio/{g}')):
        songname = f'./Datasets/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        spectral_entroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=256)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512, hop_length=256)
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=256)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256)
        zero_crossings = librosa.feature.zero_crossing_rate(y=y)

        chromagram_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        ssd_rp = rp_extract.rp_extract(wavedata=y, samplerate=sr, extract_ssd=True, extract_rp=True)
        ssd = ssd_rp.get('ssd')
        rp = ssd_rp.get('rp')

        tempo = librosa.beat.tempo(y=y, sr=sr)

        to_append += f' {np.mean(spectral_centroid)} {np.std(spectral_centroid)} {np.mean(np.diff(spectral_centroid))} {np.std(np.diff(spectral_centroid))}'
        to_append += f' {np.mean(spectral_bandwidth)} {np.std(spectral_bandwidth)} {np.mean(np.diff(spectral_bandwidth))} {np.std(np.diff(spectral_bandwidth))}'
        to_append += f' {np.mean(spectral_contrast)} {np.std(spectral_contrast)} {np.mean(np.diff(spectral_contrast))} {np.std(np.diff(spectral_contrast))}' 
        to_append += f' {np.mean(spectral_flatness)} {np.std(spectral_flatness)} {np.mean(np.diff(spectral_flatness))} {np.std(np.diff(spectral_flatness))}'
        to_append += f' {np.mean(zero_crossings)} {np.std(zero_crossings)} {np.mean(np.diff(zero_crossings))} {np.std(np.diff(zero_crossings))}'

        for mfcc in mfccs:
            to_append += f' {np.mean(mfcc)} {np.std(mfcc)} {st.skew(mfcc)} {st.kurtosis(mfcc)}'

        for item in ssd:
            to_append += f' {item}'

        for item in chromagram_cens:
            to_append += f' {np.mean(item)} {np.std(item)} {np.mean(np.diff(item))} {np.std(np.diff(item))}'
        
        for item in tonnetz:
            to_append += f' {np.mean(item)} {np.std(item)} {np.mean(np.diff(item))} {np.std(np.diff(item))}'

        for item in rp:
            to_append += f' {np.mean(item)} {np.std(item)} {st.skew(item)} {st.kurtosis(item)}'
        
        to_append += f' {tempo[0]}'

        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

