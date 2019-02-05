import pandas as pd
import numpy as np
import itertools
import sys
from sklearn import metrics, svm, model_selection, utils, pipeline, decomposition
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix, import_data_from
from reliefFsfs import ReliefFSFS

X, y, genres = import_data_from('MIR/GTZAN/SSDFeaturesGTZAN.csv')

# reliefsfs = ReliefFSFS()

# optimal_features = reliefsfs.transform(X, y)

optimal_features = pd.DataFrame(data=X)
classifier = svm.SVC(kernel='rbf', C=10, gamma='auto')
transform = decomposition.PCA(n_components=0.99)
clf = pipeline.make_pipeline((transform), (classifier))

cv_results = model_selection.cross_val_score(clf, optimal_features, y, cv=10)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(clf, optimal_features, y, cv=10)
plot_confusion_matrix(y, final_predictions, genres)