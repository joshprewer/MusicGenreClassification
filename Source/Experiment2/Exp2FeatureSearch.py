import numpy as np
import ovo_feature_selection
from sklearn import metrics, svm, model_selection, utils, multiclass, decomposition
from utilities import plot_confusion_matrix, import_data_from, cross_validation, load_xml_data
from feature_selectors import CuckooSearch, SAHS, ReliefFSFS, MRMR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import feature_selectors.binary_optimization as opt

X, y, genres = import_data_from('./Datasets/ismir04_genre/FeaturesISMIR.csv')
remaining_x = np.genfromtxt('./Datasets/ismir04_genre/RemainingFeaturesISMIR.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

clf = svm.SVC(kernel='rbf', C=3, gamma=0.02)
ovo_clf = ovo_feature_selection.OneVsOneFeatureSelection(clf)
ova_clf = ovo_feature_selection.OneVsRestClassifier(clf)

# Ovo Feature Sets
ovo_feature_sets = ovo_clf.fit(scaled_x, y)[0]
np.save('ind_ovo_fs', ovo_feature_sets)

