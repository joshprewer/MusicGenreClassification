import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
from scipy.stats import kurtosis, skew

fileName = 'ReliefFSFSMeanFeaturesGTZAN.csv'

header = 'filename' 
for i in range(0, 12):
    header += f' mfcc{i} mfcc{i}_v mfcc{i}_diff mfcc{i}_diff_v'
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
        y, sr = librosa.load(songname, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_fft=512, hop_length=256)
        to_append = f'{filename}'
        for index in range(mfcc.shape[0]):
            to_append += f' {np.mean(mfcc[index])} {np.std(mfcc[index])} {np.mean(np.diff(mfcc[index]))} {np.std(np.diff(mfcc[index]))}'
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())