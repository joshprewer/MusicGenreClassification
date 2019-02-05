import pandas as pd
import numpy as np
import itertools
import sys
from sklearn import metrics, model_selection, utils, discriminant_analysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

data = pd.read_csv('dataWithRhythm.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

classifier = discriminant_analysis.LinearDiscriminantAnalysis()
cv_results = model_selection.cross_val_score(classifier, X, y, cv=10)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(classifier, X, y, cv=10)
plot_confusion_matrix(y, final_predictions)