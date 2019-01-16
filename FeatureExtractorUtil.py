import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os
from FrFt import FrFTMFCC, frft

fileName = 'FrFTMFCCData.csv'

header = 'filename chroma_stft chroma_stftV rmse rmseV spectral_centroid spectral_centroidV spectral_bandwidth spectral_bandwidthV rolloff rolloffV zero_crossing_rate zero_crossing_rateV tempo'
for i in range(1, 14):
    header += f' mfcc{i} mfcc{i}V'
header += ' label'
header = header.split()

file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        D = np.abs(frft(y=y, exponent=0.98))**2
        S=librosa.feature.melspectrogram(S=D)
        mfcc = librosa.feature.mfcc(S=S, n_mfcc=13)
        tempo = librosa.beat.tempo(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.std(chroma_stft)} {np.mean(rmse)} {np.std(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.std(zcr)} {tempo[0]}'    
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.std(e)}'
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

