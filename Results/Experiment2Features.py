import numpy as np
from utilities import import_data_from, plot_confusion_matrix
import matplotlib.pyplot as plt
import scipy.stats

def ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def process_features_sets(data):
    feature_indexs = np.genfromtxt('./Datasets/FeatureIndexs.csv', delimiter=',').astype(int)
    feature_indexs[0] = 0
    
    hs_features = data[:, 0]
    cs_features = data[:, 1]
    dfa_features = data[:, 2]

    for (index, arr) in enumerate(hs_features):
        hs_features[index] = np.where(arr < 428)

    for (index, arr) in enumerate(cs_features):
        cs_features[index] = np.where(arr < 428)

    for (index, arr) in enumerate(dfa_features):
        dfa_features[index] = np.where(arr < 428)

    hs_feature_types = np.empty((0, 3))
    for (index, arr) in enumerate(hs_features):
        types = np.squeeze(np.take(feature_indexs, arr))
        type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
        hs_feature_types = np.vstack((hs_feature_types, type_amounts))

    cs_feature_types = np.empty((0, 3))
    for (index, arr) in enumerate(cs_features):
        types = np.squeeze(np.take(feature_indexs, arr))
        type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
        cs_feature_types = np.vstack((cs_feature_types, type_amounts))

    dfa_feature_types = np.empty((0, 3))
    for (index, arr) in enumerate(dfa_features):
        types = np.squeeze(np.take(feature_indexs, arr))
        type_amounts = np.array([len(np.where(types == 0)[0]), len(np.where(types == 1)[0]), len(np.where(types == 2)[0])])
        dfa_feature_types = np.vstack((dfa_feature_types, type_amounts))

    no_fs_feature_types = np.take(feature_indexs, np.arange(428)) 

    timbre = [len(np.where(no_fs_feature_types == 0)[0]), np.mean(hs_feature_types[:,0]), np.mean(cs_feature_types[:,0]), np.mean(dfa_feature_types[:,0])]
    tonality = [len(np.where(no_fs_feature_types == 1)[0]), np.mean(hs_feature_types[:,1]), np.mean(cs_feature_types[:,1]), np.mean(dfa_feature_types[:,1])]
    rhythm = [len(np.where(no_fs_feature_types == 2)[0]), np.mean(hs_feature_types[:,2]), np.mean(cs_feature_types[:,2]), np.mean(dfa_feature_types[:,2])]

    timbreErr = [0, ci(hs_feature_types[:,0]), ci(cs_feature_types[:,0]), ci(dfa_feature_types[:,0])]
    tonalityErr = [0, ci(hs_feature_types[:,1]), ci(cs_feature_types[:,1]), ci(dfa_feature_types[:,1])]
    rhythmErr = [0, ci(hs_feature_types[:,2]), ci(cs_feature_types[:,2]), ci(dfa_feature_types[:,2])]
    
    return [timbre, timbreErr], [tonality, tonalityErr], [rhythm, rhythmErr]

gtzan_feature_sets = np.load('./Results/Experiment2/GTZAN/ind_ovo_fs.npy') 
ismir_feature_sets = np.load('./Results/Experiment2/ISMIR/ind_ovo_fs.npy') 

ind = np.arange(4)
width = 0.5

plt.figure(1)

timbre, tonality, rhythm = process_features_sets(gtzan_feature_sets)

p1 = plt.bar(ind, timbre[0], width, yerr=timbre[1], color='r')
p2 = plt.bar(ind, tonality[0], width, bottom=timbre[0], yerr=tonality[1], color='b')
p3 = plt.bar(ind, rhythm[0], width, bottom=np.array(tonality[0])+np.array(timbre[0]), yerr=rhythm[1], color='g')

plt.title('GTZAN Feature Subsets')
plt.ylabel('Average number of features')
plt.xticks(ind, ['No FS', 'SAHS', 'BCS', 'BDFA'])
plt.legend((p1[0], p2[0], p3[0]), ('Timbre', 'Tonality', 'Rhythm'))
plt.ylim([0, 428])

plt.figure(2)

timbre, tonality, rhythm = process_features_sets(ismir_feature_sets)

p1 = plt.bar(ind, timbre[0], width, yerr=timbre[1], color='r')
p2 = plt.bar(ind, tonality[0], width, bottom=timbre[0], yerr=tonality[1], color='b')
p3 = plt.bar(ind, rhythm[0], width, bottom=np.array(tonality[0])+np.array(timbre[0]), yerr=rhythm[1], color='g')

plt.title('ISMIR Feature Subsets')
plt.ylabel('Average number of features')
plt.xticks(ind, ['No FS', 'SAHS', 'BCS', 'BDFA'])
plt.legend((p1[0], p2[0], p3[0]), ('Timbre', 'Tonality', 'Rhythm'))
plt.ylim([0, 428])

plt.show()