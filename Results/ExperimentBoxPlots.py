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

def normality_test(data):
    for result in data:
        d, p = scipy.stats.normaltest(result)
        median = np.median(result)
        print(round(p, 3))

def plot_results(x, y, yerr, title, xticks):
    plt.errorbar(x, y, yerr=yerr, fmt='x')
    plt.xticks(x, xticks)
    plt.ylim(0.55, 0.8)
    plt.title(title)
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--')


exp1_gtzan_data = [
    np.load('./Results/Experiment1/GTZAN/Scores/st_no_fs_f1.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_pca_f1.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_reliefF_f1.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_hs_f1.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_cs_f1.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_dfa_f1.npy')
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
    np.load('./Results/Experiment2/GTZAN/Scores/no_fs_f1.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/hs_f1.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/cs_f1.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/dfa_f1.npy')
]

exp2_ismir_data = [
    np.load('./Results/Experiment2/ISMIR/Scores/no_fs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/hs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/cs_f1.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/dfa_f1.npy')
]

exp1_gtzan_time = [
    np.load('./Results/Experiment1/GTZAN/Scores/st_no_fs_time.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_pca_time.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_reliefF_time.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_hs_time.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_cs_time.npy'),
    np.load('./Results/Experiment1/GTZAN/Scores/st_dfa_time.npy')
]

exp1_ismir_time = [
    np.load('./Results/Experiment1/ISMIR/Scores/st_no_fs_time.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_pca_time.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_reliefF_time.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_hs_time.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_cs_time.npy'),
    np.load('./Results/Experiment1/ISMIR/Scores/st_dfa_time.npy')
]

exp2_gtzan_time = [
    np.load('./Results/Experiment2/GTZAN/Scores/no_fs_time.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/hs_time.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/cs_time.npy'),
    np.load('./Results/Experiment2/GTZAN/Scores/dfa_time.npy')
]

exp2_ismir_timre = [
    np.load('./Results/Experiment2/ISMIR/Scores/no_fs_time.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/hs_time.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/cs_time.npy'),
    np.load('./Results/Experiment2/ISMIR/Scores/dfa_time.npy')
]


# Exp1 Results
plt.figure(1)
x, y, yerr = process_data(exp1_gtzan_data)
plot_results(x, y, yerr, 'Experiment 1: GTZAN', ['No FS', 'PCA', 'ReliefF-SFS', 'SAHS', 'BCS', 'BDFA'])

print('GTZAN Exp1 Normality Test')
normality_test(exp1_gtzan_data)

stat, pvalue = scipy.stats.f_oneway(exp1_gtzan_data[3], exp1_gtzan_data[4], exp1_gtzan_data[5])
msg = 'GTZAN Exp 1 p-value: %f' % (pvalue)
print(msg)

plt.figure(2)
x2, y2, yerr2 = process_data(exp1_ismir_data)
plot_results(x2, y2, yerr2, 'Experiment 1: ISMIR', ['No FS', 'PCA', 'ReliefF-SFS', 'SAHS', 'BCS', 'BDFA'])

print('ISMIR Exp1 Normality Test')
normality_test(exp1_ismir_data)

stat, pvalue = scipy.stats.f_oneway(exp1_ismir_data[3], exp1_ismir_data[4], exp1_ismir_data[5])
msg = 'Ismir Exp 1 p-value: %f' % (pvalue)
print(msg)

# Exp2 Results
plt.figure(3)
x3, y3, yerr3 = process_data(exp2_gtzan_data)
plot_results(x3, y3, yerr3, 'Experiment 2: GTZAN', ['No FS', 'SAHS', 'BCS', 'BDFA'])

print('GTZAN Exp2 Normality Test')
normality_test(exp2_gtzan_data)

stat, pvalue = scipy.stats.f_oneway(exp2_gtzan_data[1], exp2_gtzan_data[2], exp2_gtzan_data[3])
msg = 'GTZAN Exp 2 p-value: %f' % (pvalue)
print(msg)

plt.figure(4)
x4, y4, yerr4 = process_data(exp2_ismir_data)
plot_results(x4, y4, yerr4, 'Experiment 2: ISMIR', ['No FS', 'SAHS', 'BCS', 'BDFA'])

print('ISMIR Exp2 Normality Test')
normality_test(exp2_ismir_data)

stat, pvalue = scipy.stats.f_oneway(exp2_ismir_data[1], exp2_ismir_data[2], exp2_ismir_data[3])
msg = 'Ismir Exp 2 p-value: %f' % (pvalue)
print(msg)

plt.show()
    

    
    