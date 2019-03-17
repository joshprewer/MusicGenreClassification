import numpy as np
import ovo_feature_selection
from sklearn import metrics, svm, model_selection, utils, multiclass, decomposition
from utilities import plot_confusion_matrix, import_data_from, cross_validation, load_xml_data
from feature_selectors import CuckooSearch, SAHS, ReliefFSFS, MRMR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import feature_selectors.binary_optimization as opt

X, y, genres = import_data_from('./Datasets/ismir04_genre/FeaturesISMIR.csv')
remaining_x = np.genfromtxt('./Datasets/ismir04_genre/RemainingFeaturesISMIR.csv', delimiter=',')
X = np.hstack((X, remaining_x))

scale = MinMaxScaler((-1, 1))
scaled_x = scale.fit_transform(X)

clf = svm.SVC(kernel='rbf', C=3, gamma=0.02)
ovo_clf = multiclass.OneVsOneClassifier(clf)
ova_clf = multiclass.OneVsRestClassifier(clf)

# Feature Selectors 
n_iterations = 300

hs_ovo = SAHS.SAHS(SAHS.SAHSObjectiveFunction(scaled_x, y, harmony_menmory_size=20, hmcr_proba=0.99, iteration=n_iterations, clf=ovo_clf))
hs_ova = SAHS.SAHS(SAHS.SAHSObjectiveFunction(scaled_x, y, harmony_menmory_size=20, hmcr_proba=0.99, iteration=n_iterations, clf=ova_clf))

cs_ovo_obj = CuckooSearch.Evaluate(scaled_x, y, ovo_clf)
cs_ova_obj = CuckooSearch.Evaluate(scaled_x, y, ova_clf)

dfa_ovo_obj = CuckooSearch.Evaluate(scaled_x, y, ovo_clf)
dfa_ova_obj = CuckooSearch.Evaluate(scaled_x, y, ova_clf)

reliefF = ReliefFSFS.ReliefFSFS()

# Ovo Feature Sets
hms, hms_scores, max_score_index = hs_ovo.run()
hs_features = np.nonzero(hms[max_score_index])[0]
cs_score, g, l = opt.BCS(cs_ovo_obj, m_i=n_iterations)
cs_features = np.nonzero(g)[0]
dfa_score, g, l = opt.BDFA(dfa_ovo_obj, m_i=n_iterations)
dfa_features = np.nonzero(g)[0]

ovo_hs = hs_features
ovo_cs = cs_features
ovo_dfa = dfa_features

ovo_reliefF = reliefF.transform(scaled_x, y, ovo_clf)

np.savetxt('ovo_hs.csv', ovo_hs, delimiter=',')
np.savetxt('ovo_cs.csv', ovo_cs, delimiter=',')
np.savetxt('ovo_dfa.csv', ovo_dfa, delimiter=',')
np.savetxt('ovo_reliefF.csv', ovo_reliefF, delimiter=',')

# # # Ova Results
# hms, hms_scores, max_score_index = hs_ova.run()
# hs_features = np.nonzero(hms[max_score_index])[0]
# cs_score, g, l = opt.BCS(cs_ova_obj, m_i=n_iterations)
# cs_features = np.nonzero(g)[0]
# dfa_score, g, l = opt.BDFA(dfa_ova_obj, m_i=n_iterations)
# dfa_features = np.nonzero(g)[0]

# ova_hs = hs_features
# ova_cs = cs_features
# ova_dfa = dfa_features

# ova_reliefF = reliefF.transform(scaled_x, y, ova_clf)

# # Save results
# np.savetxt('ova_hs.csv', ova_hs, delimiter=',')
# np.savetxt('ova_cs.csv', ova_cs, delimiter=',')
# np.savetxt('ova_dfa.csv', ova_dfa, delimiter=',')
# np.savetxt('ova_reliefF.csv', ova_reliefF, delimiter=',')