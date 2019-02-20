import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import librosa
import os
from sklearn import model_selection
from utilities import plot_confusion_matrix, import_data_from
from torch.utils.data import Dataset

class GTZANDataset(Dataset):

    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx, :, :], self.Y[idx])

class LSTMClassifier(nn.Module):

	def __init__(self, input_size, embedding_dim, hidden_dim, output_size):

		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = input_size

		self.embedding = nn.Embedding(input_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

		self.hidden2out = nn.Linear(hidden_dim, output_size)
		self.softmax = nn.LogSoftmax()

		self.dropout_layer = nn.Dropout(p=0.2)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch, lengths):
		
		self.hidden = self.init_hidden(batch.size(-1))

		# embeds = self.embedding(batch)
		# packed_input = pack_padded_sequence(embeds, lengths)
		outputs, (ht, ct) = self.lstm(batch, self.hidden)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		output = self.softmax(output)

		return output


def one_hot_tensor(Y_genre_strings):
    y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genres)))
    for i, genre_string in enumerate(Y_genre_strings):
        index = genres.index(genre_string)
        y_one_hot[i, index] = 1
    return torch.from_numpy(y_one_hot)

X = np.load('LSTMFeaturesGTZAN.npy')
y = np.load('LSTMFeaturesGTZANLabels.npy')

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

batch_size = 35
nb_epochs = 400

train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.8)
train_X, val_X, train_y, val_y = model_selection.train_test_split(train_X, train_y, test_size=0.8)

train_X = torch.from_numpy(train_X)
val_X = torch.from_numpy(val_X)
test_X = torch.from_numpy(test_X)
train_y = one_hot_tensor(train_y)
val_y = one_hot_tensor(val_y)
test_y = one_hot_tensor(test_y)

trainset = GTZANDataset(train_X, train_y)

input_shape = (train_X.shape[1], train_X.shape[2])

model = LSTMClassifier(input_shape[0], input_shape[1], 32, len(genres))
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=35, shuffle=False, num_workers=0)