import pandas as pd
import numpy as np
import itertools
from sklearn import metrics, svm, model_selection, utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt
from GenreClassificationUtil import plot_confusion_matrix

cmap = plt.get_cmap('inferno')
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

data = pd.read_csv('dataWithRhythm.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

classifier = svm.SVC(kernel='rbf', gamma='auto')
cv_results = model_selection.cross_val_score(classifier, pd.DataFrame(data=X), y, cv=20)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(classifier, pd.DataFrame(data=X), y, cv=20)
plot_confusion_matrix(y, final_predictions)