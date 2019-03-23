import numpy as np
from sklearn import metrics, svm, model_selection, utils, multiclass, decomposition
from utilities import import_data_from, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

X, y, genres = import_data_from('./Datasets/ismir04_genre/FeaturesISMIR.csv')
remaining_x = np.genfromtxt('./Datasets/ismir04_genre/RemainingFeaturesISMIR.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

pca = decomposition.PCA(n_components=0.98)
reliefF_features = np.genfromtxt('./Results/Experiment1/ISMIR/FeatureSets/ovo_reliefF.csv', delimiter=',').astype(int)

hs_features = np.genfromtxt('./Results/Experiment1/ISMIR/FeatureSets/ovo_hs.csv', delimiter=',').astype(int)
cs_features = np.genfromtxt('./Results/Experiment1/ISMIR/FeatureSets/ovo_cs.csv', delimiter=',').astype(int)
dfa_features = np.genfromtxt('./Results/Experiment1/ISMIR/FeatureSets/ovo_dfa.csv', delimiter=',').astype(int)

clf = svm.SVC(kernel='rbf', C=3, gamma=0.02)
ovo_clf = multiclass.OneVsOneClassifier(clf)
number_iterations = 10

cv = len(np.unique(y))
dataset_size = len(y)
scores_array_length = cv*number_iterations

no_fs_time = np.empty(scores_array_length)
pca_time = np.empty(scores_array_length)
reliefF_time = np.empty(scores_array_length)
hs_time = np.empty(scores_array_length)
cs_time = np.empty(scores_array_length)
dfa_time = np.empty(scores_array_length)

no_fs_score = np.empty(scores_array_length)
pca_score = np.empty(scores_array_length)
reliefF_score = np.empty(scores_array_length)
hs_score = np.empty(scores_array_length)
cs_score = np.empty(scores_array_length)
dfa_score = np.empty(scores_array_length)

no_fs_f1 = np.empty(scores_array_length)
pca_f1 = np.empty(scores_array_length)
reliefF_f1 = np.empty(scores_array_length)
hs_f1 = np.empty(scores_array_length)
cs_f1 = np.empty(scores_array_length)
dfa_f1 = np.empty(scores_array_length)

correct_y = np.empty(number_iterations*dataset_size)
no_fs_y = np.empty(number_iterations*dataset_size)
pca_y = np.empty(number_iterations*dataset_size)
reliefF_y = np.empty(number_iterations*dataset_size)
hs_y = np.empty(number_iterations*dataset_size)
cs_y = np.empty(number_iterations*dataset_size)
dfa_y = np.empty(number_iterations*dataset_size)

for i in range(number_iterations):
    index = i * cv
    prediction_index = i * dataset_size
    xs, ys = utils.shuffle(scaled_x, y)

    pca_X = pca.fit_transform(xs)
    reliefF_X = np.take(xs, reliefF_features, axis=1)

    hs_X = np.take(xs, hs_features, axis=1)
    cs_X = np.take(xs, cs_features, axis=1)
    dfa_X = np.take(xs, dfa_features, axis=1)

    correct_y[prediction_index:prediction_index+dataset_size] = ys
    no_fs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, xs, ys, cv=cv)
    pca_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, pca_X, ys, cv=cv)
    reliefF_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, reliefF_X, ys, cv=cv)
    hs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, hs_X, ys, cv=cv)
    cs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, cs_X, ys, cv=cv)
    dfa_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, dfa_X, ys, cv=cv)

    no_fs_cv = model_selection.cross_validate(ovo_clf, xs, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    pca_cv = model_selection.cross_validate(ovo_clf, pca_X, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    reliefF_cv = model_selection.cross_validate(ovo_clf, reliefF_X, ys, scoring=['accuracy', 'f1_macro'], cv=cv)

    hs_cv = model_selection.cross_validate(ovo_clf, hs_X, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    cs_cv = model_selection.cross_validate(ovo_clf, cs_X, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    dfa_cv = model_selection.cross_validate(ovo_clf, dfa_X, ys, scoring=['accuracy', 'f1_macro'], cv=cv)

    no_fs_time[index:index+cv] = no_fs_cv['score_time']
    pca_time[index:index+cv] = pca_cv['score_time']
    reliefF_time[index:index+cv] = reliefF_cv['score_time']
    hs_time[index:index+cv] = hs_cv['score_time']
    cs_time[index:index+cv] = cs_cv['score_time']
    dfa_time[index:index+cv] = dfa_cv['score_time']

    no_fs_score[index:index+cv] = no_fs_cv['test_accuracy']
    pca_score[index:index+cv] = pca_cv['test_accuracy']
    reliefF_score[index:index+cv] = reliefF_cv['test_accuracy']
    hs_score[index:index+cv] = hs_cv['test_accuracy']
    cs_score[index:index+cv] = cs_cv['test_accuracy']
    dfa_score[index:index+cv] = dfa_cv['test_accuracy']

    no_fs_f1[index:index+cv] = no_fs_cv['test_f1_macro']
    pca_f1[index:index+cv] = pca_cv['test_f1_macro']
    reliefF_f1[index:index+cv] = reliefF_cv['test_f1_macro']
    hs_f1[index:index+cv] = hs_cv['test_f1_macro']
    cs_f1[index:index+cv] = cs_cv['test_f1_macro']
    dfa_f1[index:index+cv] = dfa_cv['test_f1_macro']

np.save('st_no_fs_time', no_fs_time)
np.save('st_pca_time', pca_time)
np.save('st_reliefF_time', reliefF_time)
np.save('st_hs_time', hs_time)
np.save('st_cs_time', cs_time)
np.save('st_dfa_time', dfa_time)

np.save('st_no_fs_f1', no_fs_f1)
np.save('st_pca_f1', pca_f1)
np.save('st_reliefF_f1', reliefF_f1)
np.save('st_hs_f1', hs_f1)
np.save('st_cs_f1', cs_f1)
np.save('st_dfa_f1', dfa_f1)

np.save('st_no_fs_score', no_fs_score)
np.save('st_pca_score', pca_score)
np.save('st_reliefF_score', reliefF_score)
np.save('st_hs_score', hs_score)
np.save('st_cs_score', cs_score)
np.save('st_dfa_score', dfa_score)

time_msg = "Times: %f(no_fs) %f(pca) %f(reliefF) %f(hs) %f(cs) %f(dfa)" % (no_fs_time.mean(), pca_time.mean(), reliefF_time.mean(), hs_time.mean(), cs_time.mean(), dfa_time.mean())
print(time_msg)

scores_msg = "Scores: %f(no_fs) %f(pca) %f(reliefF) %f(hs) %f(cs) %f(dfa)" % (no_fs_score.mean(), pca_score.mean(), reliefF_score.mean(), hs_score.mean(), cs_score.mean(), dfa_score.mean())
print(scores_msg)

f1_msg = "f1: %f(no_fs) %f(pca) %f(reliefF) %f(hs) %f(cs) %f(dfa)" % (no_fs_f1.mean(), pca_f1.mean(), reliefF_f1.mean(), hs_f1.mean(), cs_f1.mean(), dfa_f1.mean())
print(f1_msg)

plot_confusion_matrix(correct_y, no_fs_y, genres)
plot_confusion_matrix(correct_y, pca_y, genres)
plot_confusion_matrix(correct_y, reliefF_y, genres)
plot_confusion_matrix(correct_y, hs_y, genres)
plot_confusion_matrix(correct_y, cs_y, genres)
plot_confusion_matrix(correct_y, dfa_y, genres)