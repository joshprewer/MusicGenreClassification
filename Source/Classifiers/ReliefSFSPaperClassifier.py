import numpy as np
import sys
from sklearn import metrics, svm, model_selection, utils, pipeline, decomposition, multiclass, ensemble
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix, import_data_from
import feature_selectors.reliefFsfs 

relief = feature_selectors.reliefFsfs.ReliefFSFS()

X, y, genres = import_data_from('Datasets/GTZAN/ReliefFSFSFeaturesGTZAN.csv')
X, test = np.array_split(X, [500])
y, test = np.array_split(y, [500])
genres, test = np.array_split(genres, [5])

classifier = svm.SVC(C=50, kernel='poly', gamma=0.1)
X = relief.transform(X, y, classifier)

cv_results = model_selection.cross_val_score(classifier, X, y, cv=5)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(classifier, X, y, cv=5)
plot_confusion_matrix(y, final_predictions, genres)