import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import math
from rp_extract import rp_extract

header = 'filename'
header += ' tempo'
for i in range (0, 60):
    header += f' rh{i}'

header += ' label'
header = header.split()

fileName = 'RhythmicPatternsGTZAN.csv'
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./Datasets/GTZAN/audio/{g}'):
        print(filename)
        songname = f'./Datasets/GTZAN/audio/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        to_append = f'{filename}'

        rhythm_histogram = rp_extract.rp_extract(wavedata=y, samplerate=sr, extract_rp=True).get('rp')
        tempo = librosa.beat.tempo(y=y)[0]
        to_append +=f' {tempo}'
        for i in range(0, 60):
            to_append += f' {rhythm_histogram[i]}'
    
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())