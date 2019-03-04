import numpy as np
import matplotlib.pyplot as plt

fs_data = np.load('Results/OvoFSIndividualScores.npy')
no_fs_data = np.load('Results/OvoNoFSIndividualScores.npy')

fs_scores = fs_data[0, :, 1]
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

ind = np.arange(len(fs_scores))
width = 0.4
p1 = plt.bar(ind, fs_scores, width, color='r')
p2 = plt.bar(ind+width, no_fs_scores, width, color='b')

ax.set_title('Individual Classifer Scores')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(tick_labels)
ax.set_ylim([0, 1])
ax.autoscale_view()

plt.show()

