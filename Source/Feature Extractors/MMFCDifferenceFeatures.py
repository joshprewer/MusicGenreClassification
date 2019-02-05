import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os

fileName = 'MFCCDiff.csv'

header = 'filename' 
for i in range(1, 12):
    header += f' mfcc{i} mfcc{i}V mfcc_diff{i} mfcc_diff{i}V'
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfcc_difference = np.diff(mfcc)
        to_append = f'{filename}'
        for index in range(mfcc.shape[0]):
            to_append += f' {np.mean(mfcc[index])} {np.std(mfcc[index])} {np.mean(mfcc_difference[index])} {np.std(mfcc_difference[index])}'
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

