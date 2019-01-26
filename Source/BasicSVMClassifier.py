import pandas as pd
import numpy as np
import itertools
import sys
from sklearn import metrics, svm, model_selection, utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix, import_data_from

X, y, genres = import_data_from('MIR/GTZAN/newData.csv')

classifier = svm.SVC(kernel='rbf', C=5, gamma='auto')
cv_results = model_selection.cross_val_score(classifier, pd.DataFrame(data=X), y, cv=6)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(classifier, pd.DataFrame(data=X), y, cv=6)
plot_confusion_matrix(y, final_predictions, genres)