import pandas as pd
import numpy as np
import itertools
from sklearn import metrics, svm, model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from GenreClassificationUtil import plot_confusion_matrix

cmap = plt.get_cmap('inferno')
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

header = 'filename chroma_stft chroma_stftV rmse rmseV spectral_centroid spectral_centroidV spectral_bandwidth spectral_bandwidthV rolloff rolloffV zero_crossing_rate zero_crossing_rateV tempo'
for i in range(1, 21):
    header += f' mfcc{i} mfcc{i}V'
header += ' label'
header = header.split()

data = pd.read_csv('dataWithRhythm.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :53], dtype = float))

def dataPreProcessing():

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

	X_train = pd.DataFrame(data=X_train)
	y_train = pd.DataFrame(data=y_train)
	X_test =  pd.DataFrame(data=X_test)
	y_test = pd.DataFrame(data=y_test)

	return X_train, y_train, X_test, y_test


training_examples, training_targets, validation_examples, validation_targets = dataPreProcessing()

classifier = svm.SVC(kernel='rbf', gamma='auto')
scoring = 'accuracy'
final_predictions = classifier.fit(training_examples, training_targets).predict(validation_examples)

kfold = model_selection.KFold(n_splits=20, shuffle=True, random_state=7)
cv_results = model_selection.cross_val_score(classifier, training_examples, training_targets, cv=kfold, scoring=scoring)
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)
accuracy = metrics.accuracy_score(validation_targets, final_predictions)
print("Final accuracy (on validation data): %0.2f" % accuracy)

plot_confusion_matrix(validation_targets, final_predictions)

