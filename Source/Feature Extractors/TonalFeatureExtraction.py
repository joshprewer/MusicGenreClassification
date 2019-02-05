import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import math

header = 'filename'
for i in range(0, 12):
    header += f' chroma{i} chroma{i}_std chroma{i}_diff chroma{i}_diff_std'

for i in range(0, 6):
    header += f' tonnet{i} tonnet{i}_std tonnet{i}_diff tonnet{i}_diff_std'

header += ' label'
header = header.split()

fileName = 'TonalFeaturesGTZAN.csv'
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./MIR/GTZAN/audio/{g}'):
        print(filename)
        songname = f'./MIR/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'

        chroma_stft = librosa.feature.chroma_cens(y=y)
        for i in range(0, 12):
            to_append += f' {np.mean(chroma_stft[i])} {np.std(chroma_stft[i])} {np.mean(np.diff(chroma_stft[i]))} {np.std(np.diff(chroma_stft[i]))}'

        tonnetz = librosa.feature.tonnetz(y=y)
        for i in range(0, 6):
            to_append += f' {np.mean(tonnetz[i])} {np.std(tonnetz[i])} {np.mean(np.diff(tonnetz[i]))} {np.std(np.diff(tonnetz[i]))}'
        
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())