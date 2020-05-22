
"""
Created on Thu Feb 20 22:34:39 2020

@author: Mathias Byskov Nielsen
"""

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import read_data, clean_seqs, onehot_encoder, eight_to_three_state


####################################################
#
#   Helper functions for running the simple NN
#
####################################################

def kmerifyX(onehotX, win_size):
    """
    Creates the Kmers from the onehot-encoding and a window-size.
    
    Input:
        onehotX: One-hot encoding of features with shape(K x N x Len(Alphabet))
        win_size: Size of the sliding window
        
    Output: 
        KmersX: The Kmers of the features shape(sum(len(K)) x win_size x len(Alphabet))

    """

    middle = int(win_size / 2)
    
    kmersX = []
    zero_vec = [0 for i in range(len(onehotX[0][0]))]
    
    for i in range(len(onehotX)):        # Number of seqs
        for j in range(len(onehotX[i])): # Number of AA's in seq
            kmer = []
            
            for counter in range(win_size):
                
                if (j - middle + counter) < 0 or (j - middle + counter) > (len(onehotX[i]) - 1):
                    kmer.append(zero_vec)
                else:
                    kmer.append(onehotX[i][j - middle + counter])
            
            kmersX.append(kmer)
    
    return kmersX

def kmerifyY(onehotY):
    """
    Creates the Kmers from the onehot-encoding of the labels.
    
    Input:
        onehotY: One-hot encoding of the labels
    
    Output:
        kmersY: Kmers for the labels (sum(len(k)) x len(Alphabet))
    
    """
    
    kmersY = []
    for i in range(len(onehotY)):
        for j in range(len(onehotY[i])):
            kmersY.append(onehotY[i][j])
            
    return kmersY

class sliding_window(Dataset):
    """ Class for the kmers to input into pyTorch."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Dataframe with features and labels.
            y: 
            transform: Optional transform to be applied on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        sample_X = self.X[idx]
        sample_y = self.y[idx]
        sample = {'label': sample_y, 'features': sample_X}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def train_test_split(X, y, split = 0.20):
    """
    Create train and test samples.

    Input:


    """
    
    import random
    
    lst = [i for i in range(len(X))]
    random.shuffle(lst)
    train_amount = int((1-split)*len(X))

    train_samples_idx = lst[:train_amount]
    test_samples_idx = lst[train_amount:]
    
    X_train = [X[i] for i in train_samples_idx]
    X_test = [X[i] for i in test_samples_idx]
    
    y_train = [y[i] for i in train_samples_idx]
    y_test = [y[i] for i in test_samples_idx]
    
    return X_train, X_test, y_train, y_test

def get_accuracy(output, labels):
    """
    Computes accuracy of output-classification and labels.
    
    """
    
    correct = 0

    for idx in range(len(output)):
        if torch.argmax(output[idx]).item() == torch.argmax(labels[idx]).item():
            correct += 1
    
    acc = correct / len(output)
    return acc

def conf_matrix(output, labels):
    """
    Computes accuracy of output-classification and labels.
    
    """
    
    dct = {0:'C', 1:'H', 2:'E'}
    score_dict = {}
    
    for idx in range(len(output)):
        output_ = dct[torch.argmax(output[idx]).item()]
        label_ = dct[torch.argmax(labels[idx]).item()]
        
        string = '{}_to_{}'.format(output_, label_)
        
        if string not in score_dict:
            score_dict[string] = 1
        else:
            score_dict[string] += 1
            
    return score_dict



def flatten_features(X):
    """
    Function to flatten the dataset in correct format.

    """

    X = torch.Tensor(X)
    X = X.view([len(X),len(X[0])*len(X[0][0])]) # View like reshape in numpy

    return X

def the_nn(X, hidden_size = 110, activation_function = 'relu', dropout = 0):
    """
    Creates and fits the neural network.

    """
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(len(X[0]), hidden_size)
            self.drop_layer = nn.Dropout(p = dropout)
            self.fc2 = nn.Linear(hidden_size, 3)     
            
        def forward(self, x):
            
            if activation_function == 'relu':
                x = F.relu(self.fc1(x))
            if activation_function == 'leaky_relu':
                x = F.leaky_relu(self.fc1(x))
            if activation_function == 'tanh':
                x = F.tanh(self.fc1(x))
            if activation_function == 'sigmoid':
                x = F.sigmoid(self.fc1(x))
            
            x = self.drop_layer(x)
            x = F.softmax(self.fc2(x), dim = 1)
            
            return x
    
    net = Net()
    return net

def train(trainloader, testloader, net, criterion = nn.CrossEntropyLoss(), epochs = 50,
    optimizer = 'SGD', lr = 0.001, weight_decay = 0.01, momentum = 0):
    """
    Trains the network.
    
    """
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    
    if optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    
    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
    
    res_list = []
    for e in range(epochs):
        running_loss_train = 0
        running_accuracy_train = 0
        for sample in trainloader:
            
            features = sample['features']
            label = sample['label']

            # Training pass
            net.zero_grad()
            
            output = net(features)
            loss = criterion(output, torch.argmax(label, dim = 1))
            loss.backward()
            optimizer.step()
            
            running_loss_train += loss.item()
            running_accuracy_train += get_accuracy(output, label)
            
        with torch.no_grad():
            running_loss_test = 0
            running_accuracy_test = 0
            scoredict = {}
            for sample in testloader:
                
                features = sample['features']
                label = sample['label']
                
                output = net(features)
                loss = criterion(output, torch.argmax(label, dim = 1))
                
                running_loss_test += loss.item()
                running_accuracy_test += get_accuracy(output, label)
                
                preds = conf_matrix(output, label)
                
                for key in list(preds.keys()):
                
                    if key not in scoredict:
                        scoredict[key] = preds[key]
                    else:
                        scoredict[key] += preds[key]
                
            print(scoredict)                
        epoch_res = f"Epoch: {e + 1} - Loss: {round(running_loss_train/len(trainloader), 4)} - Accuracy: {round(running_accuracy_train/len(trainloader), 4)} - Val_Loss: {round(running_loss_test/len(testloader), 4)} - Val_accuracy: {round(running_accuracy_test/len(testloader), 4)}"
        print(epoch_res)
        
        res_list.append(epoch_res)
    
    return res_list


####################################################
#
#   Running the NN
#
####################################################

def main(path, method = 'DSSP', conversion_method = 'method_1', window_size = 17, 
         batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 20, 
         lr = 0.001, weight_decay = 0, activation_function = 'relu', momentum = 0, optimizer = 'SGD', dropout = 0):
    
    parameters = locals()
    acids_to_int = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'B': 4, 'C': 5, 'Q': 6, 'E': 7, 'Z': 8, 'G': 9, 'H': 10, 'I': 11, 'L': 12, 'K':13, 'M':14,  'F':15, 'P':16, 'S':17, 'T':18, 'W':19, 'Y':20, 'V':21}
    structures_to_int = {'C': 0, 'H': 1, 'E': 2}
    
    
    print('|------  Running Single Layer NN  ------|\n')
    
    # Read files
    print('Read and modify seqs + structures...\n')
    proteins, seqs, structures = read_data(path, method)
    seqs, proteins = clean_seqs(seqs, structures)
    
    # Onehot encode X and y
    onehotX = onehot_encoder(seqs, acids_to_int)

    structures = eight_to_three_state(structures, conversion_method)
    onehotY = onehot_encoder(structures, structures_to_int)

    # Kmerify X and y
    KmersX, KmersY = kmerifyX(onehotX, window_size), kmerifyY(onehotY)

    # Divide datasets into train and test
    X_train, X_test, y_train, y_test = train_test_split(KmersX, KmersY, split = split)

    # Initialize training and test data
    X_train, X_test = flatten_features(X_train), flatten_features(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    print(X_train.shape)
    trainloader = sliding_window(X_train, y_train, transform = None)
    trainloader = data.DataLoader(trainloader, batch_size=batch_size, shuffle = shuffle)

    testloader = sliding_window(X_test, y_test, transform = None)
    testloader = data.DataLoader(testloader, batch_size=batch_size, shuffle = shuffle)

    # Create network
    net = the_nn(X_train, hidden_size = hidden_size, activation_function=activation_function, dropout = dropout)

    # Train network
    print('Trains the network network with parameters:')
    print(parameters)
    res_list = train(trainloader, testloader, net, epochs = epochs, lr = lr, weight_decay = weight_decay, momentum = momentum, optimizer = optimizer)
    
    return str(parameters), res_list


