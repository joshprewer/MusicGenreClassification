import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import itertools
import math
import os
from librosa import util, filters
from sklearn import metrics, utils, model_selection, svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt

def import_data_from(path):

    data = pd.read_csv(path)
    data.head()

    # Dropping unneccesary columns
    data = data.drop(['filename'],axis=1)

    genre_list = data.iloc[:, -1]
    genres = np.unique(genre_list)
    encoder = LabelEncoder()

    y = encoder.fit_transform(genre_list)
    x = np.array(data.iloc[:, :-1], dtype = float)

    return x, y, genres

def load_xml_data(path, filename):
    tree = et.parse(path)
    root = tree.getroot()

    X = np.atleast_2d(np.ndarray((1000)))
    y = np.ndarray((1000), dtype='|U16')

    index = 0
    for data in root.iter('data_set'):
        sample_data = np.empty([])
        label = data.find('data_set_id')
        path = os.path.dirname(label.text)
        genre = os.path.basename(path)
        y[index] = genre

        for value in data.iter('v'):
            sample_data = np.append(sample_data, float(value.text))

        sample_data = np.delete(sample_data, 0)

        X = np.resize(X, (1000, len(sample_data)))
        X[index, :] = sample_data
        index += 1
    
    np.savetxt(f'{filename}.csv', X, delimiter=',')

    return X

def relative_correlation(weight, input_X, input_Y):
    feature_sets = np.nonzero(weight)[0]
    x = np.take(input_X, feature_sets, axis=1)
        
    k = x.shape[1]
    mutual_info = 0
    index = 0 
    for s in range(k):
        for t in range(s + 1, k):
            mutual_info += metrics.mutual_info_score(x[:, s], x[:, t])
            index += 1
    ri = mutual_info / index

    mutual_info = 0 
    index = 0
    for s in range(k):
        mutual_info += metrics.mutual_info_score(x[:, s], input_Y)
        index += 1
    rt = mutual_info / index

    rc = (k * rt) / math.sqrt(k + k * (k - 1) * ri)
    return rc

def ovo_objective_function(x, y, clf):
    cv = len(np.unique(y))
    cv_results = model_selection.cross_val_score(classifier, x, y, cv=cv)
    return cv_results.mean()

def ova_objective_function(x, y):
    cv = len(np.unique(y))
    cv_results = model_selection.cross_val_score(clf, x, y, cv=cv)
    return cv_results.mean()

def cross_validation(X, y, clf, genres):
    cv = len(np.unique(y))
    cv_results = model_selection.cross_val_score(clf, X, y, cv=cv)
    msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
    print(msg)

    final_predictions = model_selection.cross_val_predict(clf, X, y, cv=cv)
    plot_confusion_matrix(y, final_predictions, genres)

def plot_confusion_matrix(validation_targets, final_predictions, genres):

	cm = metrics.confusion_matrix(validation_targets, final_predictions)
	cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
	cmap="bone_r"
	plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
	plt.title("Confusion matrix")
	plt.xticks(np.arange(len(genres)), genres, rotation=45)
	plt.yticks(np.arange(len(genres)), genres)
	plt.ylabel("True label")
	plt.xlabel("Predicted label")

	fmt = '.2f'
	thresh = cm_normalized.max() / 2.
	for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
	    plt.text(j, i, format(cm_normalized[i, j] * 100, fmt),
	             horizontalalignment="center",
	             color="white" if cm_normalized[i, j] > thresh else "black")
	plt.show()