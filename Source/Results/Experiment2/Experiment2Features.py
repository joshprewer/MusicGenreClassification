import numpy as np
from utilities import import_data_from, plot_confusion_matrix
import matplotlib.pyplot as plt

X, y, genres = import_data_from('./Datasets/GTZAN/FeaturesGTZAN.csv')
remaining_x = np.genfromtxt('./Datasets/GTZAN/RemainingFeaturesGTZAN.csv', delimiter=',')
X = np.hstack((X, remaining_x))

feature_sets = np.load('./Source/Results/Experiment2/GTZAN/ind_ovo_fs.npy') 

hs_features = feature_sets[:, 0]
cs_features = feature_sets[:, 1]
dfa_features = feature_sets[:, 2]
feature_indexs = np.genfromtxt('./Datasets/FeatureIndexs.csv', delimiter=',').astype(int)
feature_indexs[0] = 0

for (index, arr) in enumerate(hs_features):
    hs_features[index] = np.where(arr < 428)

for (index, arr) in enumerate(cs_features):
    cs_features[index] = np.where(arr < 428)

for (index, arr) in enumerate(dfa_features):
    dfa_features[index] = np.where(arr < 428)

hs_feature_types = np.empty((45, 3))
for (index, arr) in enumerate(hs_features):
    types = np.squeeze(np.take(feature_indexs, arr))
    type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
    hs_feature_types[index, :] = type_amounts

cs_feature_types = np.empty((45, 3))
for (index, arr) in enumerate(cs_features):
    types = np.squeeze(np.take(feature_indexs, arr))
    type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
    cs_feature_types[index, :] = type_amounts

dfa_feature_types = np.empty((45, 3))
for (index, arr) in enumerate(dfa_features):
    types = np.squeeze(np.take(feature_indexs, arr))
    type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
    dfa_feature_types[index, :] = type_amounts


no_fs_feature_types = np.take(feature_indexs, np.arange(428)) 

timbre = [len(np.where(no_fs_feature_types == 0)[0]), np.mean(hs_feature_types[:,0]), np.mean(cs_feature_types[:,0]), np.mean(dfa_feature_types[:,0])]
tonality = [len(np.where(no_fs_feature_types == 1)[0]), np.mean(hs_feature_types[:,1]), np.mean(cs_feature_types[:,1]), np.mean(dfa_feature_types[:,1])]
rhythm = [len(np.where(no_fs_feature_types == 2)[0]), np.mean(hs_feature_types[:,2]), np.mean(cs_feature_types[:,2]), np.mean(dfa_feature_types[:,2])]

fig, ax = plt.subplots()
ind = np.arange(4)
width = 0.5

p1 = plt.bar(ind, timbre, width, color='r')
p2 = plt.bar(ind, tonality, width, bottom=timbre, color='b')
p3 = plt.bar(ind, rhythm, width, bottom=np.array(tonality)+np.array(timbre), color='g')

ax.set_title('Feature Subsets')
plt.ylabel('Average number of features')
plt.xticks(ind, ['No FS', 'SAHS', 'BCS', 'BDFA'])
ax.legend((p1[0], p2[0], p3[0]), ('Timbre', 'Tonality', 'Rhythm'))
ax.set_ylim([0, 428])

plt.show()