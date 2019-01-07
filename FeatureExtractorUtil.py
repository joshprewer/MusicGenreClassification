import librosa
import pandas as pd
import numpy as np
import pathlib
import csv
import os

header = 'filename chroma_stft chroma_stftV rmse rmseV spectral_centroid spectral_centroidV spectral_bandwidth spectral_bandwidthV rolloff rolloffV zero_crossing_rate zero_crossing_rateV'
for i in range(1, 21):
    header += f' mfcc{i} mfcc{i}V'
header += ' label'
header = header.split()

file = open('newData.csv', 'w', newline='')
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.std(chroma_stft)} {np.mean(rmse)} {np.std(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)}  {np.std(rolloff)} {np.mean(zcr)}  {np.std(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.std(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

