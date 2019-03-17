import scipy.stats as st
import pandas as pd
import numpy as np
import pathlib
import csv
import os
import librosa
import glob
from rp_extract import rp_extract
from mutagen.mp3 import MP3

fileName = 'FeaturesISMIR.csv'
genres = 'classical electronic jazz metal pop punk rock world'.split()

for g in genres:
    for filename in sorted(glob.iglob(f'./Datasets/ismir04_genre/audio/training/{g}/*.mp3', recursive=True)):
        filename_str = filename
        filename_str = filename_str.replace("\\", "/")
        audio = MP3(filename_str)
        length = (audio.info.length - 15) / 2
        y, sr = librosa.load(filename_str, mono=True, offset=length, duration=30)
        to_append = f'{filename}'
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)
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
