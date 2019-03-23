import numpy as np
from sklearn import metrics, svm, model_selection, utils, multiclass, decomposition
from utilities import import_data_from, plot_confusion_matrix
import ovo_multi_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

X, y, genres = import_data_from('./Datasets/ismir04_genre/FeaturesISMIR.csv')
remaining_x = np.genfromtxt('./Datasets/ismir04_genre/RemainingFeaturesISMIR.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

ovo_feature_sets = np.load('./Results/Experiment2/ISMIR/ind_ovo_fs.npy')

ovo_fs_hs = ovo_feature_sets[:, 0]
ovo_fs_cs = ovo_feature_sets[:, 1]
ovo_fs_dfa = ovo_feature_sets[:, 2]

clf = svm.SVC(kernel='rbf', C=3, gamma=0.02)

ovo_clf = multiclass.OneVsOneClassifier(clf)
ovo_clf_hs = ovo_multi_features.OneVsOneClassifier(clf, ovo_fs_hs)
ovo_clf_cs = ovo_multi_features.OneVsOneClassifier(clf, ovo_fs_cs)
ovo_clf_dfa = ovo_multi_features.OneVsOneClassifier(clf, ovo_fs_dfa)

cv = len(np.unique(y))
dataset_size = len(y)
scores_array_length = cv*10

no_fs_time = np.empty(scores_array_length)
hs_time = np.empty(scores_array_length)
cs_time = np.empty(scores_array_length)
dfa_time = np.empty(scores_array_length)

no_fs_score = np.empty(scores_array_length)
hs_score = np.empty(scores_array_length)
cs_score = np.empty(scores_array_length)
dfa_score = np.empty(scores_array_length)

no_fs_f1 = np.empty(scores_array_length)
hs_f1 = np.empty(scores_array_length)
cs_f1 = np.empty(scores_array_length)
dfa_f1 = np.empty(scores_array_length)

correct_y = np.empty(10*dataset_size)
no_fs_y = np.empty(10*dataset_size)
hs_y = np.empty(10*dataset_size)
cs_y = np.empty(10*dataset_size)
dfa_y = np.empty(10*dataset_size)

for i in range(10):
    index = i * cv
    prediction_index = i*dataset_size
    xs, ys = utils.shuffle(scaled_x, y)

    correct_y[prediction_index:prediction_index+dataset_size] = ys
    no_fs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf, xs, ys, cv=cv)
    hs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf_hs, xs, ys, cv=cv)
    cs_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf_cs, xs, ys, cv=cv)
    dfa_y[prediction_index:prediction_index+dataset_size] = model_selection.cross_val_predict(ovo_clf_dfa, xs, ys, cv=cv)

    no_fs_cv = model_selection.cross_validate(ovo_clf, xs, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    hs_cv = model_selection.cross_validate(ovo_clf_hs, xs, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    cs_cv = model_selection.cross_validate(ovo_clf_cs, xs, ys, scoring=['accuracy', 'f1_macro'], cv=cv)
    dfa_cv = model_selection.cross_validate(ovo_clf_dfa, xs, ys, scoring=['accuracy', 'f1_macro'], cv=cv)

    no_fs_time[index:index+cv] = no_fs_cv['score_time']
    hs_time[index:index+cv] = hs_cv['score_time']
    cs_time[index:index+cv] = cs_cv['score_time']
    dfa_time[index:index+cv] = dfa_cv['score_time']

    no_fs_score[index:index+cv] = no_fs_cv['test_accuracy']
    hs_score[index:index+cv] = hs_cv['test_accuracy']
    cs_score[index:index+cv] = cs_cv['test_accuracy']
    dfa_score[index:index+cv] = dfa_cv['test_accuracy']

    no_fs_f1[index:index+cv] = no_fs_cv['test_f1_macro']
    hs_f1[index:index+cv] = hs_cv['test_f1_macro']
    cs_f1[index:index+cv] = cs_cv['test_f1_macro']
    dfa_f1[index:index+cv] = dfa_cv['test_f1_macro']

np.save('no_fs_time', no_fs_time)
np.save('hs_time', hs_time)
np.save('cs_time', cs_time)
np.save('dfa_time', dfa_time)

np.save('no_fs_score', no_fs_score)
np.save('hs_score', hs_score)
np.save('cs_score', cs_score)
np.save('dfa_score', dfa_score)

np.save('no_fs_f1', no_fs_f1)
np.save('hs_f1', hs_f1)
np.save('cs_f1', cs_f1)
np.save('dfa_f1', dfa_f1)

time_msg = "Times: %f(no_fs) %f(hs) %f(cs) %f(dfa)" % (no_fs_time.mean(), hs_time.mean(), cs_time.mean(), dfa_time.mean())
print(time_msg)

scores_msg = "Scores: %f(no_fs) %f(hs) %f(cs) %f(dfa)" % (no_fs_score.mean(), hs_score.mean(), cs_score.mean(), dfa_score.mean())
print(scores_msg)

f1_msg = "f1: %f(no_fs) %f(hs) %f(cs) %f(dfa)" % (no_fs_f1.mean(), hs_f1.mean(), cs_f1.mean(), dfa_f1.mean())
print(f1_msg)

plot_confusion_matrix(correct_y, no_fs_y, genres)
plot_confusion_matrix(correct_y, hs_y, genres)
plot_confusion_matrix(correct_y, cs_y, genres)
plot_confusion_matrix(correct_y, dfa_y, genres)