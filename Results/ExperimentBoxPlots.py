import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def process_data(data):
    x = np.empty((0,1))
    y = np.empty((0,1))
    yerr = np.empty((0,1))

    for (index, element) in enumerate(data):
        mean, h = mean_confidence_interval(element)
        x = np.append(x, index)
        y = np.append(y, mean)
        yerr = np.append(yerr, h)

    return x, y, yerr

exp1_gtzan_data = [
    np.load('./Results/Experiment1/GTZAN/Scores/st_no_fs_score.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_pca_score.npy'), 
    np.load('./Results/Experiment1/GTZAN/Scores/st_reliefF_score.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_hs_score.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_cs_score.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_dfa_score.npy')
]

exp1_ismir_data = [
    np.load('./Results/Experiment1/ISMIR/Scores/st_no_fs_f1.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_pca_f1.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_reliefF_f1.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_hs_f1.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_cs_f1.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_dfa_f1.npy')
]

exp2_gtzan_data = [
    np.load('./Results/Experiment2/GTZAN/Scores/no_fs_score.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/hs_score.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/cs_score.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/dfa_score.npy')
]

exp2_ismir_data = [
    np.load('./Results/Experiment2/ISMIR/Scores/no_fs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/hs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/cs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/dfa_f1.npy')
]

plt.figure(1)

# Exp1 Results
x, y, yerr = process_data(exp1_gtzan_data)
y = y * 100
yerr = yerr * 100
plt.errorbar(x, y, yerr=yerr, fmt='x')
plt.xticks(x, ['No FS', 'PCA', 'ReliefF-SFS', 'SAHS', 'BCS', 'BDFA'])
plt.ylim(55, 84)
plt.title('Experiment 1: GTZAN')
plt.ylabel('Classification accuracy (%)')
plt.grid(True, linestyle='--')

stat, pvalue = scipy.stats.f_oneway(exp1_gtzan_data[0], exp1_gtzan_data[1], exp1_gtzan_data[3], exp1_gtzan_data[4], exp1_gtzan_data[5])
msg = 'GTZAN Exp 1 p-value: %f' % (pvalue)
print(msg)

plt.figure(2)
x2, y2, yerr2 = process_data(exp1_ismir_data)
plt.errorbar(x2, y2, yerr=yerr2, fmt='x')
plt.xticks(x, ['No FS', 'PCA', 'ReliefF-SFS', 'SAHS', 'BCS', 'BDFA'])
plt.ylim(0.55, 0.84)
plt.title('Experiment 1: ISMIR')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--')

stat, pvalue = scipy.stats.f_oneway(exp1_ismir_data[0], exp1_ismir_data[1], exp1_ismir_data[3], exp1_ismir_data[4], exp1_ismir_data[5])
msg = 'Ismir Exp 1 p-value: %f' % (pvalue)
print(msg)

# Exp2 Results
plt.figure(3)
x3, y3, yerr3 = process_data(exp2_gtzan_data)
y3 = y3 * 100
yerr3 = yerr3 * 100
plt.errorbar(x3, y3, yerr=yerr3, fmt='x')
plt.xticks(x3, ['No FS', 'SAHS', 'BCS', 'BDFA'])
plt.ylim(55, 84)
plt.title('Experiment 2: GTZAN')
plt.ylabel('Classification accuracy (%)')
plt.grid(True, linestyle='--')

stat, pvalue = scipy.stats.f_oneway(exp2_gtzan_data[0], exp2_gtzan_data[1], exp2_gtzan_data[2], exp2_gtzan_data[3])
msg = 'GTZAN Exp 2 p-value: %f' % (pvalue)
print(msg)

plt.figure(4)
x4, y4, yerr4 = process_data(exp2_ismir_data)
plt.errorbar(x4, y4, yerr=yerr4, fmt='x')
plt.xticks(x4, ['No FS', 'SAHS', 'BCS', 'BDFA'])
plt.ylim(0.55, 0.84)
plt.title('Experiment 2: ISMIR')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--')

stat, pvalue = scipy.stats.f_oneway(exp2_ismir_data[0], exp2_ismir_data[1], exp2_ismir_data[2], exp2_ismir_data[3])
msg = 'Ismir Exp 2 p-value: %f' % (pvalue)
print(msg)

plt.show()
    

    
    