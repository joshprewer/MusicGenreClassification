import essentia
from essentia.standard import *
import scipy.stats as st
import pandas as pd
import numpy as np
import pathlib
import librosa
import csv
import os
import sys
from rp_extract import rp_extract

fileName = 'Datasets/GTZAN/SharmaFeaturesGTZAN.csv'
header = 'filename'

class feature_vector:
    def __init__(self, name):
        self.mean = f'{name}'
        self.variance = f'{name}_var'
        self.median= f'{name}_med'
        self.skewness = f'{name}_skew'
        self.kurtosis = f'{name}_kurt'
        self.min = f'{name}_min'
        self.max = f'{name}_max'

        self.data = None

    def get_features(self):
        return f' {np.mean(self.data)} {np.std(self.data)} {np.median(self.data)} {st.skew(self.data)} {st.kurtosis(self.data)} {np.min(self.data)} {np.max(self.data)}'

mfcc_features = []
for i in range(0, 13):
    feature = feature_vector(f'mfcc{i}')
    mfcc_features = np.append(mfcc_features, feature)
    header += f' {feature.mean}'

ssd_features = []
for i in range(0, 24):
    feature = feature_vector(f'ssd{i}')
    ssd_features = np.append(ssd_features, feature)
    header += f' {feature.mean} {feature.variance} {feature.median} {feature.skewness} {feature.kurtosis} {feature.min} {feature.max}'

rhythm_features = []
for i in range (0, 60):
    header += f' rh{i}'

spectral_centroid_features = []
spectral_rolloff_features = []
zcr_features = []
for i in range(0, 26):
    centroid = feature_vector(f'centroid{i}')
    rolloff = feature_vector(f'rolloff{i}')
    zcr = feature_vector(f'zcr{i}')
    header += f' {centroid.mean} {rolloff.mean} {zcr.mean}'

chroma_features = []
for i in range(0, 12):
    feature = feature_vector(f'chroma{i}')
    header += f' {feature.mean}'

header += ' rms bpm label'
header = header.split()
    
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

spectrum = Spectrum()

barkbands = BarkBands(sampleRate=22050)
centroidClass = SpectralCentroidTime(sampleRate=22050)
rolloffClass = RollOff(sampleRate=22050)
zcrClass = ZeroCrossingRate()
rmsClass = RMS()

for g in genres:
    for filename in os.listdir(f'./Datasets/GTZAN/audio/{g}'):
        songname = f'./Datasets/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)

        for i in range (0, 13):
            to_append += f' {np.mean(mfcc[i])}'

        rhythm_histogram = rp_extract.rp_extract(wavedata=y, samplerate=sr, extract_rh=True).get('rh')
        ssd = rp_extract.rp_extract(wavedata=y, samplerate=sr, extract_ssd=True).get('ssd')
        
        for item in ssd:
            to_append += f' {item}'

        for i in range (0, 60):
            to_append += f' {rhythm_histogram[i]}'
    

        frameCount = 0
        for frame in FrameGenerator(y, frameSize=(22050*5), hopSize=22050, startFromZero=True):
            if frameCount >= 26:
                break 
            frameCount += 1
            centroid = centroidClass(frame)
            rolloff = rolloffClass(spectrum(frame))
            zcr = zcrClass(frame)

            to_append += f' {centroid} {rolloff} {zcr}'

        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(0, 12):
            to_append += f' {np.mean(chromagram[i])}'
            
        rms = rmsClass(y)
        bpm = librosa.beat.tempo(y=y, sr=sr)

        to_append += f' {rms} {bpm[0]} {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

