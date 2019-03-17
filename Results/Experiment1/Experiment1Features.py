import numpy as np
from utilities import import_data_from, plot_confusion_matrix
from sklearn import decomposition, utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def process_feature_sets(data):
    
    timbre = np.empty((0,1))
    tonality = np.empty((0,1))
    rhythm = np.empty((0,1))

    for (index, element) in enumerate(data):
        feature_types = np.squeeze(np.take(feature_indexs, np.where(element < max_n_features)))

        timbre = np.append(timbre, len(np.where(feature_types == 0)[0]))
        tonality = np.append(tonality, len(np.where(feature_types == 1)[0]))
        rhythm = np.append(rhythm, len(np.where(feature_types == 2)[0]))

    timbre = np.insert(timbre, 1, 0)
    tonality = np.insert(tonality, 1, 0)
    rhythm = np.insert(rhythm, 1, 0)

    return timbre, tonality, rhythm

max_n_features = 428

feature_indexs = np.genfromtxt('./Datasets/FeatureIndexs.csv', delimiter=',').astype(int)
feature_indexs[0] = 0
 
gtzan_data = [
    np.arange(max_n_features),
    np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_reliefF.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_hs.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_cs.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/GTZAN/ovo_dfa.csv', delimiter=',').astype(int)
]

ismir_data = [
    np.arange(max_n_features),
    np.genfromtxt('./Source/Results/Experiment1/ISMIR/ovo_reliefF.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/ISMIR/ovo_hs.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/ISMIR/ovo_cs.csv', delimiter=',').astype(int),
    np.genfromtxt('./Source/Results/Experiment1/ISMIR/ovo_dfa.csv', delimiter=',').astype(int)
]

gtzan_X, gtzan_y, gtzan_genres = import_data_from('./Datasets/GTZAN/FeaturesGTZAN.csv')
gtzan_remaining_x = np.genfromtxt('./Datasets/GTZAN/RemainingFeaturesGTZAN.csv', delimiter=',')
gtzan_X = np.hstack((gtzan_X, gtzan_remaining_x))

scale = MinMaxScaler((-1, 1))
gtzan_scaled_x = scale.fit_transform(gtzan_X)
gtzan_xs, gtzan_ys = utils.shuffle(gtzan_scaled_x, gtzan_y)
gtzan_pca = decomposition.PCA(n_components=0.98).fit_transform(gtzan_xs).shape[1]
pca = np.array([0, gtzan_pca, 0, 0, 0, 0])

width = 0.5

timbre, tonality, rhythm = process_feature_sets(gtzan_data)

plt.figure(1)
ind = np.arange(6)

p1 = plt.bar(ind, timbre, width, color='r')
p2 = plt.bar(ind, tonality, width, bottom=timbre, color='b')
p3 = plt.bar(ind, rhythm, width, bottom=np.array(tonality)+np.array(timbre), color='g')
p4 = plt.bar(ind, pca, width, bottom=np.array(tonality)+np.array(timbre)+np.array(rhythm), color='grey')

plt.title('GTZAN Feature Subsets')
plt.ylabel('Number of features')
plt.xticks(ind, ['No FS', 'PCA', 'ReliefF', 'SAHS', 'BCS', 'BDFA'])
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Timbre', 'Tonality', 'Rhythm', 'PCA'))
plt.ylim([0, max_n_features])

timbre, tonality, rhythm = process_feature_sets(ismir_data)

ismir_X, y, genres = import_data_from('./Datasets/ismir04_genre/FeaturesISMIR.csv')
ismir_remaining_x = np.genfromtxt('./Datasets/ismir04_genre/RemainingFeaturesISMIR.csv', delimiter=',')
ismir_X = np.hstack((ismir_X, ismir_remaining_x))
ismir_scaled_x = scale.fit_transform(ismir_X)
ismir_xs, y = utils.shuffle(ismir_scaled_x, y)
ismir_pca = decomposition.PCA(n_components=0.98).fit_transform(ismir_xs).shape[1]
pca = np.array([0, ismir_pca, 0, 0, 0, 0])

plt.figure(2)

p1 = plt.bar(ind, timbre, width, color='r')
p2 = plt.bar(ind, tonality, width, bottom=timbre, color='b')
p3 = plt.bar(ind, rhythm, width, bottom=np.array(tonality)+np.array(timbre), color='g')
p4 = plt.bar(ind, pca, width, bottom=np.array(tonality)+np.array(timbre)+np.array(rhythm), color='grey')

plt.title('ISMIR Feature Subsets')
plt.ylabel('Number of features')
plt.xticks(ind, ['No FS', 'PCA', 'ReliefF', 'SAHS', 'BCS', 'BDFA'])
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Timbre', 'Tonality', 'Rhythm', 'PCA'))
plt.ylim([0, max_n_features])
plt.show()