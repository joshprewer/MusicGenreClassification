import numpy as np
import matplotlib.pyplot as plt

score_data = np.load('Results/OvOIndividualScores.npy')
labels = np.load('Results/OvONoFSIndividualScores.npy')[1, :]

sahs_feature_index = np.genfromtxt('Datasets/GTZAN/SAHSFeaturesIndex.csv', delimiter=",")
sahs2_feature_index = np.genfromtxt('Datasets/GTZAN/SAHSRemainingFeatureIndex.csv', delimiter=",")
ssd_feature_index = np.genfromtxt('Datasets/GTZAN/SSD+RPFeaturesIndex.csv', delimiter=",")

sahs_feature_index[0] = 2.0
sahs2_feature_index[0] = 1.0
ssd_feature_index = np.delete(ssd_feature_index, 0)

feature_index = np.concatenate((sahs_feature_index, sahs2_feature_index, ssd_feature_index))

cs_features = np.empty((score_data.shape[0], 5)) 
hs_features = np.empty((score_data.shape[0], 5)) 

for idx, item in enumerate(score_data):
    cs_feature_set = item[1]
    cs_feature_types = np.take(feature_index, cs_feature_set)
    cs_features[idx] = [cs_feature_types[cs_feature_types == 0].size, cs_feature_types[cs_feature_types == 1].size,
                        cs_feature_types[cs_feature_types == 2].size, cs_feature_types[cs_feature_types == 3].size, 
                        cs_feature_types[cs_feature_types == 4].size] 

    hs_feature_set = item[1]
    hs_feature_types = np.take(feature_index, hs_feature_set)
    hs_features[idx] = [hs_feature_types[hs_feature_types == 0].size, hs_feature_types[hs_feature_types == 1].size,
                        hs_feature_types[hs_feature_types == 2].size, hs_feature_types[hs_feature_types == 3].size, 
                        hs_feature_types[hs_feature_types == 4].size] 

tick_labels = np.empty(labels.shape, dtype='|U16')
label_dict =  {
    0: 'B',
    1: 'Cl',
    2: 'Co',
    3: 'D',
    4: 'Hh',
    5: 'J',
    6: 'M',
    7: 'P',
    8: 'Re',
    9: 'Ro'
}

for idx, item in enumerate(labels):
    tick_labels[idx] = f'{label_dict[item[0]]}v{label_dict[item[1]]}'

features_to_test = hs_features

intensity = features_to_test[:, 0]
pitch = features_to_test[:, 1]
timbre = features_to_test[:, 2]
tonality = features_to_test[:, 3]
rhythm = features_to_test[:, 4]

fig, ax = plt.subplots()

ind = np.arange(features_to_test.shape[0])
width = 0.15
p1 = plt.bar(ind, intensity, width, color='r')
p2 = plt.bar(ind+width, pitch, width, color='b')
p3 = plt.bar(ind+width*2, timbre, width, color='g')
p4 = plt.bar(ind+width*3, tonality, width, color='yellow')
p5 = plt.bar(ind+width*4, rhythm, width, color='orange')

ax.set_title('Individual Classifer Feature Sets')
ax.set_xticks(ind + width / 5)
ax.set_xticklabels(tick_labels, fontsize=7)
ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Intensity', 'Pitch', 'Timbre', 'Tonality', 'Timbre'))
ax.autoscale_view()

plt.show()

