import essentia
from essentia.standard import *
import scipy.stats as st
import pandas as pd
import numpy as np
import pathlib
import csv
import os

fileName = 'SSDFeaturesGTZAN.csv'

class feature_vector:
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
    ssd_features = np.append(ssd_features, feature_vector(f'ssd{i}'))

header = 'filename'

for feature in ssd_features:
    header += f' {feature.mean} {feature.variance} {feature.median} {feature.skewness} {feature.kurtosis} {feature.min} {feature.max}'

header += ' label'
header = header.split()
    
file = open(fileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

spectrum = Spectrum()
w = Windowing(type = 'hann')

for g in genres:
    for filename in os.listdir(f'./MIR/GTZAN/audio/{g}'):
        songname = f'./MIR/GTZAN/audio/{g}/{filename}'
        loader = essentia.standard.MonoLoader(filename=songname)
        y = loader()

        ssd = []
        bark_bands = None
        barkbands = BarkBands()

        for frame in FrameGenerator(y, frameSize=1024, hopSize=512, startFromZero=True):
            ssd.append(barkbands(spectrum(w(frame))))

        ssd = essentia.array(ssd).T
        to_append = f'{filename}'

        for i in range(0, 24):
            ssd_features[i].data = ssd[i]
            to_append += ssd_features[i].get_features()
        
        to_append += f' {g}'
        file = open(fileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
