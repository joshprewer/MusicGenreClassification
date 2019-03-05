import numpy as np
import matplotlib.pyplot as plt

score_data = np.load('Results/OvoIndividualScores.npy')
labels = np.load('Results/OvoNoFSIndividualScores.npy')[1, :]

scores = np.empty((score_data.shape[0], 3)) 

for idx, item in enumerate(score_data):
    scores[idx] = np.asarray([item[0], item[2], item[4]])

cs_scores = scores[:, 0]
hs_scores = scores[:, 1]
no_fs_scores = scores[:, 2]

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
ax.set_xticklabels(tick_labels, fontsize=7)
ax.set_ylim([0, 1])
ax.legend((p1[0], p2[0], p3[0]), ('Cuckoo Search', 'Self-Adapting Harmony Search', 'No FS'))
ax.autoscale_view()

plt.show()