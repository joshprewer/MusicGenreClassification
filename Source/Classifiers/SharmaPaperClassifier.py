import numpy as np
import itertools
from sklearn import metrics, svm, model_selection, utils, pipeline, decomposition, multiclass, ensemble
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix, import_data_from

paperX, y, genres = import_data_from('SharmaFeaturesGTZAN.csv')

# classifier = svm.SVC(C=15, kernel='rbf', gamma=0.001, tol=1e-2, decision_function_shape='ovo')
# classifier = svm.SVC(C=100, kernel='poly', degree=1, gamma=0.00008, tol=1e-3, decision_function_shape='ovo')
classifier = svm.LinearSVC(C=0.05, max_iter=10000)
# ecoc = multiclass.OutputCodeClassifier(classifier, code_size=4)
clf = multiclass.OneVsOneClassifier(classifier)
# X = relief.transform(paperX, y, classifier)

cv_results = model_selection.cross_val_score(clf, paperX, y, cv=)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(clf, paperX, y, cv=10)
plot_confusion_matrix(y, final_predictions, genres)