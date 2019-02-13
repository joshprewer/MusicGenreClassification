from sklearn import metrics, svm, model_selection, multiclass, ensemble
from utilities import plot_confusion_matrix, import_data_from

X, y, genres = import_data_from('Datasets/GTZAN/SharmaFeaturesGTZAN.csv')

classifier = svm.LinearSVC(C=0.012, max_iter=10000)
clf = multiclass.OneVsOneClassifier(classifier)

cv_results = model_selection.cross_val_score(clf, X, y, cv=10)
msg = "Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

final_predictions = model_selection.cross_val_predict(clf, X, y, cv=10)
plot_confusion_matrix(y, final_predictions, genres)