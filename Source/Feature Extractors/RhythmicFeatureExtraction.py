import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import math

header = 'filename'
header += ' tempo'
for i in range(0, 40):
    header += f' tempogram{i} tempogram{i}_std'

# for i in range(0, 6):
#     header += f' tonnet{i} tonnet{i}_std tonnet{i}_diff tonnet{i}_diff_std'

header += ' label'
header = header.split()

fileName = 'RhythmicFeaturesGTZAN.csv'
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

        tempogram = librosa.feature.tempogram(y=y)
        tempo = librosa.beat.tempo(y=y)[0]
        to_append +=f' {tempo}'
        for i in range(0, 40):
            to_append += f' {np.mean(tempogram[i])} {np.std(tempogram[i])}'

        # tonnetz = librosa.feature.tonnetz(y=y)
        # for i in range(0, 6):
        #     to_append += f' {np.mean(tonnetz[i])} {np.std(tonnetz[i])} {np.mean(np.diff(tonnetz[i]))} {np.std(np.diff(tonnetz[i]))}'
        
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())