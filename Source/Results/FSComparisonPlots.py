import numpy as np
from sklearn import metrics, svm, model_selection, utils, multiclass, decomposition
from utilities import import_data_from
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

X, y, genres = import_data_from('./Datasets/GTZAN/FeaturesGTZAN.csv')
remaining_x = np.genfromtxt('./Datasets/GTZAN/RemainingFeaturesGTZAN.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

xs, ys = utils.shuffle(scaled_x, y)

pca = decomposition.PCA(n_components=0.98)

reliefF_features = np.genfromtxt('./Source/Results/ovo_reliefF.csv', delimiter=',').astype(int)

hs_features = np.genfromtxt('./Source/Results/ovo_hs.csv', delimiter=',').astype(int)
cs_features = np.genfromtxt('./Source/Results/ovo_cs.csv', delimiter=',').astype(int)
dfa_features = np.genfromtxt('./Source/Results/ovo_dfa.csv', delimiter=',').astype(int)

pca_X = pca.fit_transform(xs)
reliefF_X = np.take(xs, reliefF_features, axis=1)

hs_X = np.take(xs, hs_features, axis=1)
cs_X = np.take(xs, cs_features, axis=1)
dfa_X = np.take(xs, dfa_features, axis=1)

clf = svm.SVC(kernel='rbf', C=3, gamma=0.02)
ovo_clf = multiclass.OneVsOneClassifier(clf)
ova_clf = multiclass.OneVsRestClassifier(clf)

scores = model_selection.cross_validate(clf, xs, ys, cv=10)
pca_scores = model_selection.cross_validate(clf, pca_X, ys, cv=10)
reliefF_scores = model_selection.cross_validate(clf, reliefF_X, ys, cv=10)

hs_scores = model_selection.cross_validate(clf, hs_X, ys, cv=10)
cs_scores = model_selection.cross_validate(clf, cs_X, ys, cv=10)
dfa_scores = model_selection.cross_validate(clf, dfa_X, ys, cv=10)

fig, ax = plt.subplots()
ind = np.arange(6)

means = [scores['test_score'].mean(), pca_scores['test_score'].mean(), reliefF_scores['test_score'].mean(), hs_scores['test_score'].mean(), cs_scores['test_score'].mean(), dfa_scores['test_score'].mean()]
error = [scores['test_score'].std(), pca_scores['test_score'].std(), reliefF_scores['test_score'].std(), hs_scores['test_score'].std(), cs_scores['test_score'].std(), dfa_scores['test_score'].std()]
fit_time = [scores['fit_time'].mean(), pca_scores['fit_time'].mean(), reliefF_scores['fit_time'].mean(), hs_scores['fit_time'].mean(), cs_scores['fit_time'].mean(), dfa_scores['fit_time'].mean()]
score_time = [scores['score_time'].mean(), pca_scores['score_time'].mean(), reliefF_scores['score_time'].mean(), hs_scores['score_time'].mean(), cs_scores['score_time'].mean(), dfa_scores['score_time'].mean()]

times = np.add(fit_time, score_time)

_means = np.array(means) 
_error = np.array(error) 

b1 = plt.bar(ind, _means, yerr=_error)

plt.xticks(ind, ('No FS', 'PCA', 'ReliefF-SFS', 'HS', 'CS', 'DFA'))
plt.ylim(0.70, 0.85)

plt.show()