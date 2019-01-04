import pandas as pd
import numpy as np
import itertools
from sklearn import metrics, svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import cm, gridspec, pyplot as plt

cmap = plt.get_cmap('inferno')
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
	X = scaler.fit_transform(np.array(data.iloc[:, 6:-1], dtype = float))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	X_train = pd.DataFrame(data=X_train)
	y_train = pd.DataFrame(data=y_train)
	X_test =  pd.DataFrame(data=X_test)
	y_test = pd.DataFrame(data=y_test)

	return X_train, y_train, X_test, y_test


def plotConfusionMatrix(validation_targets, final_predictions):

	cm = metrics.confusion_matrix(validation_targets, final_predictions)
	# Normalize the confusion matrix by row (i.e by the number of samples
	# in each class).
	cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
	cmap="bone_r"
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title("Confusion matrix")
	plt.xticks(np.arange(10), genres, rotation=45)
	plt.yticks(np.arange(10), genres)
	plt.ylabel("True label")
	plt.xlabel("Predicted label")

	fmt = '.2f'
	thresh = cm_normalized.max() / 2.
	for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
	    plt.text(j, i, format(cm_normalized[i, j] * 100, fmt),
	             horizontalalignment="center",
	             color="white" if cm_normalized[i, j] > thresh else "black")
	plt.show()


training_examples, training_targets, validation_examples, validation_targets = dataPreProcessing()

classifier = svm.SVC(kernel='linear')
final_predictions = classifier.fit(training_examples, training_targets).predict(validation_examples)
accuracy = metrics.accuracy_score(validation_targets, final_predictions)
print("Final accuracy (on validation data): %0.2f" % accuracy)

plotConfusionMatrix(validation_targets, final_predictions)

