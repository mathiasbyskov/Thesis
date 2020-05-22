
"""
Created on Thu Mar 12 23:43:41 2020

@author: Mathias Byskov Nielsen

"""

import os
os.chdir('C:/Users/mathi/Desktop/Thesis/Implementations')

import torch
import numpy as np
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import Dataset


from utils import read_data, clean_seqs, onehot_encoder, eight_to_three_state

############################################################
#
#   Helper functions for running the RNN
#
############################################################

def pad_input(sequences):
    """  Helper function: Pad sequences to correct format. """
    from copy import deepcopy
    
    # Deepcopy list
    padded_sequences = deepcopy(sequences)
    
    # Define length of alphabet (int)
    len_alphabet = len(padded_sequences[0][0])
    
    # Find max length of a seq (int)
    max_len = 0
    for seq1 in padded_sequences:
        if len(seq1) > max_len:
            max_len = len(seq1)
    
    # Define vector to append (list)
    pad = [0 for _ in range(len_alphabet)]
    
    # Add padding to all sequences
    for idx, seq in enumerate(padded_sequences):
        length = max_len - len(seq)
        for _ in range(length):
            padded_sequences[idx].append(pad)

    return padded_sequences

def int_encode_Y(structures, vocab_dict):
    """ Helper function: Integer encode structures {0, 1, 2} """
    import numpy as np
    
    int_structures = []
    
    # Convert to integers
    for structure in structures:
            int_structures.append([vocab_dict[char] for char in structure])
            
    # Find max length
    max_len = 0
    for structure in int_structures:
        if len(structure) > max_len:
            max_len = len(structure)
    
    # Pad seqeucens with ''
    for idx, structure in enumerate(int_structures):
        for _ in range(max_len + 1):
            if _ > len(structure):
                int_structures[idx].append(np.nan)
        
    return int_structures

class sample_loader(Dataset):
    """ Class for the samples to input into pyTorch."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X: 
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
        sample = {'target': sample_y, 'sequence': sample_X}
        
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
    train_amount = int((1 - split) * len(X))

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
        if torch.argmax(output[idx]).item() == labels[idx].item():
            correct += 1
    
    acc = correct / len(output)
    return acc

def the_nn(input_size, hidden_size, n_layers, output_size, layer_type = 'RNN', nonlinearity = 'tanh', dropout = 0, bidirectional = False):    
    """
    Creates and fits the recurrent neural network.
    
    """
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            
            # Defining layers
            self.hidden_size = hidden_size
            self.n_layers = n_layers
        
            # RNN Layer
            if layer_type == 'RNN':
                self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = n_layers, batch_first = True, 
                                  bidirectional = bidirectional, nonlinearity = nonlinearity, dropout = dropout)   
            
            if layer_type == 'LSTM':
                self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = n_layers, batch_first = True, 
                                  bidirectional = bidirectional, dropout = dropout)
            
            if layer_type == 'GRU':
                self.rnn = nn.GRU(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = n_layers, batch_first = True, 
                                  bidirectional = bidirectional, dropout = dropout)
            
            if bidirectional == False:
                self.fc = nn.Linear(hidden_size, output_size)
            
            if bidirectional == True:
                self.fc = nn.Linear(hidden_size*2, output_size)
            
        def forward(self, x):
            batch_size = x.size(0)
            
            #Initializing hidden state for first input using method defined below
            hidden = self.init_hidden(batch_size, self.hidden_size)
                
            # Find sequence lengths (for packing)
            x_lengths = self.find_lengths(x)
            
            # Pack sequences
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

            # Run the network
            out, hidden = self.rnn(x, hidden)
            
            # Unpack the sequences again
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # Run through the linear layer
            out = F.relu(self.fc(out))
            
            # Perform log_softmax on output (WORSE PERFORMANCE!)
            #x = F.log_softmax(x, dim = 2)

            return out, hidden
            
        def init_hidden(self, batch_size, hidden_size):
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            
            if layer_type == 'RNN' or layer_type == 'GRU':
                
                if bidirectional == False:
                    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
                    
                if bidirectional == True:
                    hidden = torch.zeros(2, batch_size, self.hidden_size)
            
            if  layer_type == 'LSTM':
                
                if bidirectional == False:
                    hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                              torch.zeros(self.n_layers, batch_size, self.hidden_size))
                    
                if bidirectional == True:
                    hidden = (torch.zeros(2, batch_size, self.hidden_size),
                              torch.zeros(2, batch_size, self.hidden_size))
                    
            return hidden
        
        def find_lengths(self, input_seq):
            # Find seq-lengths of each sequence (used to pack sequences)
            x_lengths = []
            for seq in input_seq:
                for idx, vec in enumerate(seq):
                    if sum(vec).item() != 1:
                        x_lengths.append(idx)
                        break
                    if idx == 752:
                        x_lengths.append(len(seq))   
            return x_lengths
        
    net = Model()
    return net

def the_nn_article(input_size, dropout):    
    """
    Creates and fits the recurrent neural network from the article.
    
    """
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            
            # Defining layers
            self.hidden_size = 256
            self.first_layer = 512
            self.second_layer = 1024
            self.n_layers = 2
            self.bidirectional = True
            self.dropout = dropout
        
            # RNN Layer
            self.rnn = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, 
                                  num_layers = self.n_layers, batch_first = True, 
                                  bidirectional = self.bidirectional, dropout = self.dropout)
            
            self.fc1 = nn.Linear(self.first_layer, self.second_layer)
            self.fc2 = nn.Linear(self.second_layer, 3)
            
        def forward(self, x):
            batch_size = x.size(0)
            
            #Initializing hidden state for first input using method defined below
            hidden = self.init_hidden(batch_size, self.hidden_size)
                
            # Find sequence lengths (for packing)
            x_lengths = self.find_lengths(x)
            
            # Pack sequences
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

            # Run the network
            out, hidden = self.rnn(x, hidden)
            
            # Unpack the sequences again
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # Run through the linear layer
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            
            # Perform log_softmax on output (WORSE PERFORMANCE!)
            #x = F.log_softmax(x, dim = 2)

            return out, hidden
            
        def init_hidden(self, batch_size, hidden_size):
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            
            hidden = (torch.zeros(2*self.n_layers, batch_size, self.hidden_size),
                      torch.zeros(2*self.n_layers, batch_size, self.hidden_size))
                    
            return hidden
        
        def find_lengths(self, input_seq):
            # Find seq-lengths of each sequence (used to pack sequences)
            x_lengths = []
            for seq in input_seq:
                for idx, vec in enumerate(seq):
                    if sum(vec).item() != 1:
                        x_lengths.append(idx)
                        break
                    if idx == 752:
                        x_lengths.append(len(seq))   
            return x_lengths
        
    net = Model()
    return net

def train(trainloader, testloader, net, criterion = nn.CrossEntropyLoss(), num_classes = 3, 
          epochs = 20, optimizer = 'SGD', lr = 0.01, weight_decay = 0, momentum = 0):
    """
    Trains the network.
    
    """
        
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    
    if optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    
    res_list = []
    for epoch in range(epochs):
        running_loss_train = 0
        running_accuracy_train = 0
        for sample in trainloader:
            
            # Define input and target
            input_seq = sample['sequence']
            target_seq = sample['target']

            # Clear existing gradients
            optimizer.zero_grad() 

            # Runs the RNN
            output, hidden = net(input_seq)

            # Slice target-seq (to fit with max_length in batch)
            target_seq = target_seq.narrow(1, 0, output.shape[1]).contiguous().view(-1).long()

            # Get mask (values that have been padded)
            mask = (target_seq > -1)
            
            # Flatten output
            output = output.contiguous().view(-1, num_classes)
            
            # Filter values in target_seq and output
            target_seq = target_seq[mask]
            output = output[mask, :]

            # Calculate loss and update weights accordingly
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step() 

            running_loss_train += loss.item()
            running_accuracy_train += get_accuracy(output, target_seq)
        
        with torch.no_grad():
            running_loss_test = 0
            running_accuracy_test = 0
            for sample in testloader:
            
                input_seq = sample['sequence']
                target_seq = sample['target']

                output, _ = net(input_seq)
                
                target_seq = target_seq.narrow(1, 0, output.shape[1]).contiguous().view(-1).long()
                mask = (target_seq > -1)
                output = output.contiguous().view(-1, num_classes)
                target_seq = target_seq[mask]
                output = output[mask, :]
                
                loss = criterion(output, target_seq)
                
                running_loss_test += loss.item()
                running_accuracy_test += get_accuracy(output, target_seq)
                
        epoch_res = f"Epoch: {epoch + 1} - Loss: {round(running_loss_train/len(trainloader), 4)} - Accuracy: {round(running_accuracy_train/len(trainloader), 4)} - Val_Loss: {round(running_loss_test/len(testloader), 4)} - Val_accuracy: {round(running_accuracy_test/len(testloader), 4)}"
        print(epoch_res)
        
        res_list.append(epoch_res)
    
    return res_list

####################################################
#
#   Running the RNN
#
####################################################

def main(path, article = False, method = 'DSSP', conversion_method = 'method_1', batch_size = 50, 
         split = 0.2, shuffle = True, epochs = 20, lr = 0.01,  optimizer = 'SGD',
         layer_type = 'RNN', nonlinearity = 'tanh', dropout = 0,
         hidden_size = 20, weight_decay = 0, momentum = 0, bidirectional = False):
    
    print('\n|------  Running RNN  ------|\n')
    print('Read and modify seqs + structures...\n')
    
    parameters = locals()
    
    # Define int-encoding
    acids_to_int = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'B': 4, 'C': 5, 'Q': 6, 'E': 7, 'Z': 8, 'G': 9, 'H': 10, 'I': 11, 'L': 12, 'K':13, 'M':14, 'F':15, 'P':16, 'S':17, 'T':18, 'W':19, 'Y':20, 'V':21}
    structures_to_int = {'C': 0, 'H': 1, 'E': 2}
        
    # Read files
    files, seqs, structures = read_data(path, method)
    seqs, proteins = clean_seqs(seqs, structures)
        
    # Create features (X) and targets (Y)
    onehotX = onehot_encoder(seqs, acids_to_int)
    padded_X = pad_input(onehotX)
        
    structures = eight_to_three_state(structures, conversion_method)
    int_Y = int_encode_Y(structures, structures_to_int)
        
    # Create train and test loader
    X_train, X_test, y_train, y_test = train_test_split(padded_X, int_Y, split = split)
    
    train_loader = sample_loader(torch.Tensor(np.array(X_train, dtype=float)), torch.Tensor(y_train), transform = None)
    train_loader = data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle)
    
    test_loader = sample_loader(torch.Tensor(np.array(X_test, dtype=float)), torch.Tensor(y_test), transform = None)
    test_loader = data.DataLoader(test_loader, batch_size=batch_size, shuffle = shuffle)
        
    # Instantiate the model with hyperparameters
    input_size = len(acids_to_int)
    output_size = len(structures_to_int)
    
    if article == False:
        net = the_nn(input_size = input_size, hidden_size = hidden_size, n_layers = 1, 
                     output_size = output_size, bidirectional = bidirectional, 
                     layer_type = layer_type, nonlinearity = nonlinearity, dropout = dropout)
    
    if article == True:
        net = the_nn_article(input_size, dropout = dropout)
    
    # Train the RNN
    res = train(train_loader, test_loader, net, criterion = nn.CrossEntropyLoss(), epochs = epochs, 
              optimizer = optimizer, lr = lr, weight_decay = weight_decay, momentum = momentum)

    return str(parameters), res




