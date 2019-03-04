import numpy as np
import matplotlib.pyplot as plt

cs_data = np.load('Results/OvoCSFSIndividualScores.npy')
hs_data = np.load('Results/OvoFSIndividualScores.npy')
no_fs_data = np.load('Results/OvoNoFSIndividualScores.npy')

cs_scores = cs_data[0, :]
hs_scores = hs_data[0, :, 1]
no_fs_scores = no_fs_data[0, :]
labels = no_fs_data[1, :]

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

index = 0
for item in labels:
    tick_labels[index] = f'{label_dict[item[0]]}v{label_dict[item[1]]}'
    index += 1

fig, ax = plt.subplots()

ind = np.arange(len(hs_scores))
width = 0.2
p1 = plt.bar(ind, cs_scores, width, color='r')
p2 = plt.bar(ind+width, hs_scores, width, color='b')
p3 = plt.bar(ind+width+width, no_fs_scores, width, color='g')

ax.set_title('Individual Classifer Scores')
ax.set_xticks(ind + width / 3)
ax.set_xticklabels(tick_labels)
ax.set_ylim([0, 1])
ax.autoscale_view()

plt.show()

