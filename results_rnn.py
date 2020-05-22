
import os
os.chdir('C:/Users/mathi/Desktop/Thesis/Implementations/')

from rnn import main

path = '../DATA/CB513_formatted.txt'

#%%

# vanilla-rnn

f = open("../Results/rnn/vailla-rnn.txt", "w")
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:

    parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                           bidirectional = False, layer_type = 'RNN', hidden_size = 3, 
                           lr = lr, optimizer = 'SGD', nonlinearity = 'tanh', epochs = 50)
    
    f.write("Training for VANILLA-RNN. lr: {} \n".format(lr))        
    f.write(parameters + '\n')
    
    for r in res:
        f.write(r + '\n')
            
    f.write('\n\n')
f.close()

#%%

# Linear Layer (100 hidden units)

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/rnn-linear-layer.txt", "w")
learning_rates = [0.001, 0.01, 0.1]


for lr in learning_rates:
    parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                           bidirectional = False, layer_type = 'RNN', hidden_size = 100, 
                           lr = lr, optimizer = 'SGD', nonlinearity = 'tanh', epochs = 50)
    
    f.write("Training for RNN-LINEAR-LAYER. lr: {} \n".format(lr))        
    f.write(parameters + '\n')
    
    for r in res:
        f.write(r + '\n')
            
    f.write('\n\n')
f.close()

#%%

# Bidirectional!

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/rnn-linear-layer-bidirectional.txt", "w")



parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                       bidirectional = True, layer_type = 'RNN', hidden_size = 100, 
                       lr = 0.1, optimizer = 'SGD', nonlinearity = 'tanh', epochs = 50)
    
f.write("Training for RNN-LINEAR-LAYER. BIDIRECTIONAL. \n")        
f.write(parameters + '\n')
    
for r in res:
    f.write(r + '\n')
            
f.write('\n\n')
f.close()

#%%

# Gated units! (LSTM & GRU)

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/LSTM-GRU-BRNN.txt", "w")
layer_type = ['LSTM', 'GRU']

for typ in layer_type:
    parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                           bidirectional = True, layer_type = typ, hidden_size = 100, 
                           lr = 0.1, optimizer = 'SGD', nonlinearity = 'tanh', epochs = 50)
    
    f.write("Training for RNN-LINEAR-LAYER. type: {} \n".format(typ))        
    f.write(parameters + '\n')
    
    for r in res:
        f.write(r + '\n')
            
    f.write('\n\n')
f.close()

#%%

# Gated units 2! (LSTM & GRU)

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/LSTM-GRU-BRNN-2.txt", "w")
layer_type = ['LSTM', 'GRU']

for typ in layer_type:
    parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                           bidirectional = True, layer_type = typ, hidden_size = 100, 
                           lr = 0.001, optimizer = 'Adam', nonlinearity = 'tanh', epochs = 50)
    
    f.write("Training for RNN-LINEAR-LAYER. type: {} \n".format(typ))        
    f.write(parameters + '\n')
    
    for r in res:
        f.write(r + '\n')
            
    f.write('\n\n')
f.close()
#%%

batch_size = [10, 50]
learning_rates = [0.01, 0.001]
optimizer = ['RMSprop', 'Adam']
momentum = [0.1, 0.01, 0.001]
layer_type = ['LSTM']

nonlinearity = 'relu'
hidden_size = 100

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/LSTM-GRU-BRNN-3.txt", "w")


num_round = 0
for bs in batch_size:
    for lr in learning_rates:
        for optim in optimizer:
            for mom in momentum:
                for lt in layer_type:
                    num_round += 1
                    
                    parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', batch_size = bs, 
                                           split = 0.2, shuffle = True, epochs = 50, lr = lr,  optimizer = optim,
                                           layer_type = lt, nonlinearity = nonlinearity, dropout = 0,
                                           hidden_size = hidden_size, weight_decay = 0, momentum = mom, bidirectional = True)
                    f.write(parameters + '\n')
    
                    for r in res:
                        f.write(r + '\n')
                    f.write('\n\n')
                    print('Round: {} out of 48 done!'.format(num_round))
f.close()           



#%%

main(path, method = 'DSSP', conversion_method = 'method_2', batch_size = 50, 
         split = 0.2, shuffle = True, epochs = 20, lr = 0.01,  optimizer = 'SGD',
         layer_type = 'RNN', nonlinearity = 'tanh', dropout = 0,
         hidden_size = 20, weight_decay = 0, momentum = 0, bidirectional = False)

# GOOD RESULTS (BUT SLOWLY)
# Epoch: 18 - Loss: 0.7241 - Accuracy: 0.6873 - Val_Loss: 0.7554 - Val_accuracy: 0.6777

main(path, method = 'DSSP', conversion_method = 'method_2', bidirectional = True, layer_type = 'LSTM',
     hidden_size = 256, lr = 0.01, optimizer = 'Adam', nonlinearity = 'relu', epochs = 50)



#%%

# MODEL FROM PAPER

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/rnn-paper-copy.txt", "w")



parameters, res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                       bidirectional = True, lr = 0.01, optimizer = 'Adam', epochs = 100)
    
f.write("Training for RNN-LINEAR-LAYER. BIDIRECTIONAL. \n")        
f.write(parameters + '\n')
    
for r in res:
    f.write(r + '\n')
            
f.write('\n\n')
f.close()


#%%

import os
os.chdir('C:/Users/mathi/Desktop/Thesis/Implementations/')

from rnn import main

# MODEL FROM PAPER

path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/rnn-paper-copy--1.txt", "w")



parameters, res = main(path, article = True, method = 'DSSP', conversion_method = 'method_2', 
                       bidirectional = True, lr = 0.001, optimizer = 'Adam', epochs = 80, dropout = 0)
    
f.write("Training for RNN-LINEAR-LAYER. BIDIRECTIONAL. \n")        
f.write(parameters + '\n')
    
for r in res:
    f.write(r + '\n')
            
f.write('\n\n')
f.close()


path = '../DATA/CB513_formatted.txt'
f = open("../Results/rnn/rnn-paper-copy--2.txt", "w")



parameters, res = main(path, article = True, method = 'DSSP', conversion_method = 'method_2', 
                       bidirectional = True, lr = 0.001, optimizer = 'Adam', epochs = 80, dropout = 0.5)
    
f.write("Training for RNN-LINEAR-LAYER. BIDIRECTIONAL. \n")        
f.write(parameters + '\n')
    
for r in res:
    f.write(r + '\n')
            
f.write('\n\n')
f.close()
