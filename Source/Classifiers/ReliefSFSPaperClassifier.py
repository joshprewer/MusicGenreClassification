import numpy as np
import sys
from sklearn import metrics, svm, model_selection, utils, pipeline, decomposition, multiclass, ensemble
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix, import_data_from
from reliefFsfs import *

relief = ReliefFSFS()
paperX, y, genres = import_data_from('ReliefFSFSFeaturesGTZAN.csv')
paperX, test = np.array_split(paperX, [500])
y, test = np.array_split(y, [500])
genres, test = np.array_split(genres, [5])

classifier = svm.SVC(C=50, kernel='poly', gamma=0.1)

cv_results = model_selection.cross_val_score(classifier, paperX, y, cv=5)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(classifier, paperX, y, cv=5)
plot_confusion_matrix(y, final_predictions, genres)