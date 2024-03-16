import os
import pandas as pd
import numpy as np
import scipy
import glob
import math
import matplotlib.pyplot as plt
from operator import *
import gc

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from scipy.signal import butter, lfilter

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Filtering functions
def butter_bandpass(lowcut, highcut, fs, order=6):
  return butter(order, [lowcut, highcut], fs=fs, btype='band')
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y
def calcsnr(prefilter, filtered):
  filteredsum = 0
  denomsum = 0
  for i in range(len(prefilter)):
    filteredsum = filteredsum+filtered[i]*filtered[i]
    denomsum = denomsum+ (filtered[i]-prefilter[i])*(filtered[i]-prefilter[i])
  return 10*math.log10(filteredsum/denomsum)

# Custom dataset object
class EEGDataset(Dataset):
  def __init__(self, eeglist, labels, transform=None, target_transform=None):
    self.labels = torch.from_numpy(np.array(labels))
    self.labels = self.labels.to(torch.float32)
    self.eeglist = eeglist
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):

    label = self.labels[idx]
    raw = self.eeglist[idx]
    eeg = torch.from_numpy(raw.get_data())
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return eeg, label

# Collect Raw objects from MNE in Python lists
data_list = {"raw":[], "preprocessed":[], "manual_preprocess":[], "rawsample": [], "preprocessedsample":[], "manual_preprocesssample":[]}
labels = []

# Get subject information
readtsv = pd.read_csv('participants.tsv', sep = '\t')

# Channel names, taken from OpenNeuro
channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

# Parse in .set files
for i in range(1,89):

  filteredpath = r"C:\Users\togar\Documents\UofT\Spring Term 2024\APS360 - Fundamentals of Deep Learning\EEG-Alzheimer-Detection\OpenNeuro Preprocessed/sub-%s/eeg"%(str(i).zfill(3))
  filteredfile = glob.glob(os.path.join(filteredpath, '*.set'))
  filtered = mne.io.read_raw_eeglab(filteredfile[0], preload = True)

  rawpath = r"C:\Users\togar\Documents\UofT\Spring Term 2024\APS360 - Fundamentals of Deep Learning\EEG-Alzheimer-Detection\Raw Data/sub-%s/eeg"%(str(i).zfill(3))
  file = glob.glob(os.path.join(rawpath, '*.set'))
  raw = mne.io.read_raw_eeglab(file[0], preload = True)

  # Update data lists
  data_list['raw'].append(raw)
  data_list['preprocessed'].append(filtered)

  label = readtsv['Group'][i-1]
  # Change labels to onehot encodings
  if label == 'A':
    labels.append([0,0,1])
  elif label == 'F':
    labels.append([0,1,0])
  elif label == 'C':
    labels.append([1,0,0])



from mne_icalabel import label_components
# Exclude noisy artifacts
exclusion = ['line noise', 'heartbeat', 'eye blink']

def manual_process(raw, plotting = True):
    freq_low = 0.5
    freq_high = 45
    iirparams = dict(order = 4, ftype = 'butter')

    raw.filter(freq_low, freq_high, method = 'iir', iir_params = iirparams)

    # Create ICA object and fit it to the data
    ica = mne.preprocessing.ICA(n_components=19, random_state=97, verbose = False)
    ica.fit(raw)

    # Plot ICA components to identify artifacts
    icalabels = label_components(raw, ica, method = 'iclabel')
    if plotting == True:
        ica.plot_components()
        picks = list(range(0,18))
        ica.plot_properties(raw, picks=picks)
        print(icalabels)
    # Exclude bad channels
    ica.exclude = []
    for i, label in enumerate(icalabels):
        if label in exclusion:
            ica.exclude.append(i)
    ica.apply(raw)

    return raw




# from IPython.display import clear_output
# i = 0
# for subject in data_list['raw']:
#     raw = subject.load_data()
#     data_list['manual_preprocess'].append(manual_process(raw, plotting = False))
#     clear_output()
#     print(i)
#     i+=1
    
    
import gc
gc.collect()

trainlist, vallist, testlist, trainlabels, vallabels, testlabels = [],[],[],[],[],[]
labelonehot = [[0,0,1], [0,1,0], [1,0,0]]
#make sure enough of each group in each partition of data
for label in labelonehot:
  count = 0
  for i in range (len(labels)):
    if labels[i] == label:
      if (countOf(labels, label)-count)>8:
        n = data_list['raw'][i].copy()
        trainlist.append(n.crop(0,20))
        trainlabels.append(labels[i])
      if (countOf(labels, label)-count)>3:
        n = data_list['raw'][i].copy()
        vallist.append(n.crop(0,20))
        vallabels.append(labels[i])
      else:
        n = data_list['raw'][i].copy()
        testlist.append(n.crop(0,20))
        testlabels.append(labels[i])
      count = count + 1
      gc.collect()


trainset = EEGDataset(trainlist, trainlabels)
valset = EEGDataset(vallist, vallabels)
testset = EEGDataset(testlist, testlabels)


trainloader = DataLoader(trainset, batch_size = 1, shuffle = True)
valloader = DataLoader(valset, batch_size = 1, shuffle = True)
testloader = DataLoader(testset, batch_size = 1, shuffle = True)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class CNN_EEG_Classifier(nn.Module):
    def __init__(self):
        super(CNN_EEG_Classifier, self).__init__()
        #USE LARGER KERNEL BECAUSE IMAGES ARE HIGH RESOLUTION, FEATURES TAKE UP MORE PIXELS
        self.conv1 = nn.Conv1d(19, 25, 2) 
        
        #BATCH NORMALIZATION TO PREVENT VANISHING GRADIENTS
        self.bn1 = nn.BatchNorm1d(25)
        #LARGE POOLING KERNEL TO REDUCE DIMENSIONALIZATION FASTER (JUST FOUND THIS TO BE HELPFUL BY EXPERIMENTING)
        self.pool = nn.MaxPool1d(4, 4)

        self.conv2 = nn.Conv1d(25, 30, 2) 

        self.bn2 = nn.BatchNorm1d(30)
        self.fc1 = nn.Linear(18720, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):
        batch_size = x.shape[0]
        outconv1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        outconv2 = self.pool(F.relu(self.bn2(self.conv2(outconv1))))

        outconv2 = outconv2.view(batch_size, -1)

        outfc1 = F.relu(self.fc1(outconv2))
        # print(outfc1.shape)

        outfc2 = self.fc2(outfc1)
        # print(outfc2.shape)

        out = F.softmax(outfc2, dim = 1)

        return out



class CNN_LSTM_EEG(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_EEG, self).__init__()
        # This is taking in 64 channels from what I understand, we could change it to 19
        self.conv1 = nn.Conv1d(19, 64, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten() # Rad removed this line
        self.lstm = nn.LSTM(64, 100, 1, batch_first = True)
        self.dropout2 = nn.Dropout(0.5)


        self.dense1 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(100, 50)
        self.dropout4 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = (self.pool(self.dropout1(F.relu(self.conv2(x)))))
        x = self.flatten(x)
        x, (a, b) = self.lstm(x)
        x = self.dropout2(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(F.relu(x))
        x = self.dropout4(x)
        x = self.dense3(F.sigmoid(x))

        return x
    

import torch.optim as optim
import time

def get_accuracy(model, train=True, train_data = trainloader, val_data = valloader):
    if train:
        dataloader = train_data
    else:
        dataloader = val_data
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            outputs = model(imgs)
            predicted = torch.argmax(outputs, dim=1)
            _, labels = torch.max(labels, dim=1)
            # print(predicted, labels)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total
torch.manual_seed(5)
def train(model,batch_size=1, traindata = trainloader, valdata = valloader, num_epochs=10, rate = 0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(traindata):

            # print(imgs.shape)
            imgs = imgs.to(torch.float32)
            out = model(imgs)             # forward pass
            # print(out.shape)
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)

            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, train=True, train_data=traindata)) # compute training accuracy
            val_acc.append(get_accuracy(model, train=False, val_data = valdata))  # compute validation accuracy
            n += 1
            print("Epoch:", epoch)
            print("Training Accuracy:", get_accuracy(model, train=True, train_data=traindata))
            print("Validation Accuracy:", get_accuracy(model, train=False, val_data = valdata))
            
    print("Iterations:", n)
    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')

    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

eegclassifier = CNN_LSTM_EEG(3)
# eegclassifier = CNN_EEG_Classifier()
train(eegclassifier)


print("Test accuracy:",get_accuracy(eegclassifier,train = True, train_data = testloader))