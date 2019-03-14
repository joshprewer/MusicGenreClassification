import pandas as pd
import numpy as np
from sklearn import metrics, svm, model_selection, utils, pipeline, decomposition, multiclass, mixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utilities import plot_confusion_matrix, import_data_from, cross_validation, load_xml_data
import ovo_feature_selection
import ovo_multi_features
import os
from feature_selectors import CuckooSearch, SAHS, ReliefFSFS
from skfeature.function.information_theoretical_based import MRMR

# X = np.genfromtxt('Test.csv', delimiter=',')
# remaining_x = np.genfromtxt('Datasets/GTZAN/NewSAHSRemaining.csv', delimiter=',')
# other_remaining_x, y, genres = import_data_from('Datasets/GTZAN/SSD+RPFeaturesGTZAN.csv')
# X = np.hstack((X, remaining_x))
# X = np.hstack((X, other_remaining_x))
X, y, genres = import_data_from('FeaturesGTZAN.csv')
remaining_x = np.genfromtxt('RemainingFeaturesGTZAN.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

test = MRMR.mrmr(scaled_x, y)

param_grid = [
  {'C': [0.25, 0.5, 1, 2, 4, 8, 16], 'gamma': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2]},
]

xs, ys = utils.shuffle(scaled_x, y)
classifier = svm.SVC(kernel='rbf', C=3, gamma=0.02)
# feature_sets = ovo_feature_selection.OneVsOneFeatureSelection(classifier).fit(xs, ys)

# feature_sets = np.asarray(feature_sets)
# feature_sets = np.asarray(feature_sets[0, :])

# clf = ovo_multi_features.OneVsOneClassifier(classifier, feature_sets)
# clf = multiclass.OneVsOneClassifier(classifier)

# gs = model_selection.GridSearchCV(classifier, param_grid, cv=10)
# clf = gs.fit(xs, ys)

cross_validation(xs, ys, classifier, genres)