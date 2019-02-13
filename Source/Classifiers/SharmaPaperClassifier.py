from sklearn import metrics, svm, model_selection, multiclass, ensemble
from utilities import plot_confusion_matrix, import_data_from, cross_validation

X, y, genres = import_data_from('Datasets/GTZAN/SharmaFeaturesGTZAN.csv')

classifier = svm.LinearSVC(C=0.012, max_iter=10000)
clf = multiclass.OneVsOneClassifier(classifier)

cross_validation(X, y, clf, genres)