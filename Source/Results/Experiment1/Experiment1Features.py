import numpy as np
from utilities import import_data_from, plot_confusion_matrix
import matplotlib.pyplot as plt

X, y, genres = import_data_from('./Datasets/GTZAN/FeaturesGTZAN.csv')
remaining_x = np.genfromtxt('./Datasets/GTZAN/RemainingFeaturesGTZAN.csv', delimiter=',')
X = np.hstack((X, remaining_x))

reliefF_features = np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_reliefF.csv', delimiter=',').astype(int)
hs_features = np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_hs.csv', delimiter=',').astype(int)
cs_features = np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_cs.csv', delimiter=',').astype(int)
dfa_features = np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_dfa.csv', delimiter=',').astype(int)
feature_indexs = np.genfromtxt('./Datasets/FeatureIndexs.csv', delimiter=',').astype(int)
feature_indexs[0] = 0

no_fs_feature_types = np.take(feature_indexs, np.arange(428))  
reliefF_feature_types = np.squeeze(np.take(feature_indexs, np.where(reliefF_features < 428)))
hs_feature_types = np.squeeze(np.take(feature_indexs, np.where(hs_features < 428)))
cs_feature_types = np.squeeze(np.take(feature_indexs, np.where(cs_features < 428)))
dfa_feature_types = np.squeeze(np.take(feature_indexs, np.where(dfa_features < 428)))

timbre = [len(np.where(no_fs_feature_types == 0)[0]), len(np.where(reliefF_feature_types == 0)[0]), len(np.where(hs_feature_types == 0)[0]), len(np.where(cs_feature_types == 0)[0]), len(np.where(dfa_feature_types == 0)[0])]
tonality = [len(np.where(no_fs_feature_types == 1)[0]), len(np.where(reliefF_feature_types == 1)[0]), len(np.where(hs_feature_types == 1)[0]), len(np.where(cs_feature_types == 1)[0]), len(np.where(dfa_feature_types == 1)[0])]
rhythm = [len(np.where(no_fs_feature_types == 2)[0]), len(np.where(reliefF_feature_types == 2)[0]), len(np.where(hs_feature_types == 2)[0]), len(np.where(cs_feature_types == 2)[0]), len(np.where(dfa_feature_types == 2)[0])]

fig, ax = plt.subplots()
ind = np.arange(5)
width = 0.5

p1 = plt.bar(ind, timbre, width, color='r')
p2 = plt.bar(ind, tonality, width, bottom=timbre, color='b')
p3 = plt.bar(ind, rhythm, width, bottom=np.array(tonality)+np.array(timbre), color='g')

ax.set_title('Feature Subsets')
plt.ylabel('Average number of features')
ax.set_xticklabels(['', 'No FS', 'ReliefF', 'SAHS', 'BCS', 'BDFA'])
ax.legend((p1[0], p2[0], p3[0]), ('Timbre', 'Tonality', 'Rhythm'))
ax.set_ylim([0, 428])

plt.show()