import pandas as pd
import numpy as np
import itertools
import sys
from sklearn import metrics, mixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import cm, gridspec, pyplot as plt
from utilities import plot_confusion_matrix

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

def dataPreProcessing():

	header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
	for i in range(1, 21):
	    header += f' mfcc{i}'
	header += ' label'
	header = header.split()

	data = pd.read_csv('data.csv')
	data.head()

	# Dropping unneccesary columns
	data = data.drop(['filename'],axis=1)

	genre_list = data.iloc[:, -1]
	encoder = LabelEncoder()
	y = encoder.fit_transform(genre_list)

	scaler = StandardScaler()
	X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	X_train = pd.DataFrame(data=X_train)
	y_train = pd.DataFrame(data=y_train)
	X_test =  pd.DataFrame(data=X_test)
	y_test = pd.DataFrame(data=y_test)

	return X_train, y_train, X_test, y_test


training_examples, training_targets, validation_examples, validation_targets = dataPreProcessing()
n_classes = len(np.unique(training_targets))
print(n_classes)
classifier = mixture.GaussianMixture(n_components=n_classes)
final_predictions = classifier.fit(training_examples, training_targets).predict(validation_examples)
accuracy = metrics.accuracy_score(validation_targets, final_predictions)
print("Final accuracy (on validation data): %0.2f" % accuracy)

plot_confusion_matrix(validation_targets, final_predictions, genres)

