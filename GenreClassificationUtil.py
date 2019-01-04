import numpy as np
import itertools
from sklearn import metrics
from matplotlib import cm, gridspec, pyplot as plt

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

def plot_confusion_matrix(validation_targets, final_predictions):

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