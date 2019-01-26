import numpy as np
import pandas as pd
import itertools
import math as math
from librosa import util, filters
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import cm, gridspec, pyplot as plt

def import_data_from(path):

    data = pd.read_csv(path)
    data.head()

    # Dropping unneccesary columns
    data = data.drop(['filename'],axis=1)

    genre_list = data.iloc[:, -1]
    genres = np.unique(genre_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    return x, y, genres


def fast_fractional_fourier_transform(vec, exponent):
    # Compute Fourier transform powers of vec.
    f0 = np.array(vec)
    f1 = np.fft.fft(f0, axis=0)
    f2 = negate_permutation(f0)
    f3 = negate_permutation(f1)

    # Derive eigenbasis vectors from vec's Fourier transform powers.
    b0 = f0 + f1 + f2 + f3
    b1 = f0 + 1j*f1 - f2 - 1j*f3
    b2 = f0 - f1 + f2 - f3
    b3 = f0 - 1j*f1 - f2 + 1j*f3
    # Note: vec == (b0 + b1 + b2 + b3) / 4

    # Phase eigenbasis vectors by their eigenvalues raised to the exponent.
    b1 *= 1j**exponent
    b2 *= 1j**(2*exponent)
    b3 *= 1j**(3*exponent)

    # Recombine transformed basis vectors to get transformed vec.
    return (b0 + b1 + b2 + b3) / 4


def negate_permutation(vec):
    """Returns the result of applying an FFT to the given vector twice."""
    head, tail = vec[:1], vec[1:]
    return np.concatenate((head, tail[::-1]))

    import itertools
from sklearn import metrics
from matplotlib import cm, gridspec, pyplot as plt

def plot_confusion_matrix(validation_targets, final_predictions, genres):

	cm = metrics.confusion_matrix(validation_targets, final_predictions)
	# Normalize the confusion matrix by row (i.e by the number of samples
	# in each class).
	cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
	cmap="bone_r"
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title("Confusion matrix")
	plt.xticks(np.arange(len(genres)), genres, rotation=45)
	plt.yticks(np.arange(len(genres)), genres)
	plt.ylabel("True label")
	plt.xlabel("Predicted label")

	fmt = '.2f'
	thresh = cm_normalized.max() / 2.
	for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
	    plt.text(j, i, format(cm_normalized[i, j] * 100, fmt),
	             horizontalalignment="center",
	             color="white" if cm_normalized[i, j] > thresh else "black")
	plt.show()

def stfrft(y, exponent, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = filters.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')


    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))    



    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        stft_matrix[:, bl_s:bl_t] = fast_fractional_fourier_transform(fft_window*y_frames[:, bl_s:bl_t], exponent=exponent)[:stft_matrix.shape[0]]


    return stft_matrix