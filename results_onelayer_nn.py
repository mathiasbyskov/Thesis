# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 03:33:15 2020

@author: mathi
"""

from onelayer_nn import main

path = '../DATA/CB513_formatted.txt'

parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = 0.1, weight_decay = 0, activation_function='relu', momentum = 0, optimizer = 'SGD',
                                        dropout = 0)


#%% 

# METHOD & CONVERSION METHOD

methods = ['DSSP', 'STRIDE', 'DEFINE']
conversion_methods = ['method_1', 'method_2', 'method_3', 'method_4', 'method_5']

f = open("../Results/results_methods_convmethods.txt", "w")

for method in methods:
    for conversion_method in conversion_methods:
        
        f.write("Training for {}, {}.\n".format(method, conversion_method))
        parameters, epoch_res = main(path, method = method, conversion_method = conversion_method, 
                                     lr = 0.1, epochs = 50, weight_decay = 0)
        
        f.write(parameters + '\n')
        for res in epoch_res:
            f.write(res + '\n')
        
        f.write('\n\n')
            

f.close()

#%%

# WINDOW SIZE

window_sizes = [5, 9, 13, 17, 21, 25]

f = open("../Results/results_window_size.txt", "w")

for window_size in window_sizes:
    f.write("Training for {}.\n".format(window_size))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                                    lr = 0.1, epochs = 50, weight_decay = 0, window_size = window_size)
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')
            

f.close()

#%%

# HIDDEN UNITS

hidden_units = [50, 70, 90, 110, 130, 150]

f = open("../Results/results_hidden_units.txt", "w")

for units in hidden_units:
    f.write("Training for {}.\n".format(units))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                                    lr = 0.1, epochs = 50, weight_decay = 0, window_size = 17, hidden_size = units)
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')
            

f.close()

#%%

# ACTIVATION FUNCTIONS

activation_functions = ['leaky_relu', 'tanh', 'sigmoid', 'relu']

f = open("../Results/results_activation_functions.txt", "w")

for func in activation_functions:
    f.write("Training for {}.\n".format(func))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', 
                                    lr = 0.1, epochs = 50, weight_decay = 0, window_size = 17, hidden_size = 110, activation_function=func)
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')


f.close()

#%%

# OPTIMIZERS + LEARNING RATE

optimizers = ['SGD', 'RMSprop', 'Adam']
learning_rates = [0.3, 0.1, 0.05, 0.01, 0.001]


f = open("../Results/optimizers_learning_rate.txt", "w")

for optimizer in optimizers:
    for lr in learning_rates:
        
        f.write("Training for {}, {}.\n".format(optimizer, lr))
        parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = lr, weight_decay = 0, activation_function='relu', momentum = 0, optimizer = optimizer)
        f.write(parameters + '\n')
        for res in epoch_res:
            f.write(res + '\n')
            
        f.write('\n\n')

f.close()

#%%

# OPTIMIZERS + BATCH SIZE

optimizers = ['SGD', 'RMSprop', 'Adam']
batch_sizes = [10, 50, 100, 150, 200]


f = open("../Results/optimizers_batch_size.txt", "w")

for optimizer in optimizers:
    for bs in batch_sizes:
        
        f.write("Training for {}, {}.\n".format(optimizer, bs))
        parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = bs, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = 0.1, weight_decay = 0, activation_function='relu', momentum = 0, optimizer = optimizer)
        f.write(parameters + '\n')
        for res in epoch_res:
            f.write(res + '\n')
            
        f.write('\n\n')

f.close()



#%%

# WEIGHT DECAY (L2)

weight_decays = [0, 0.001, 0.05, 0.01, 0.1, 0.3]

f = open("../Results/results_weight_decay.txt", "w")

for weight_decay in weight_decays:
    f.write("Training for {}.\n".format(weight_decay))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = 0.1, weight_decay = weight_decay, activation_function='relu', momentum = 0, optimizer = 'SGD')
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')
            

f.close()
#%%

dropouts = [0, 0.1, 0.20, 0.30, 0.5, 0.7]

f = open("../Results/results_dropout.txt", "w")

for dropout in dropouts:
    f.write("Training for {}.\n".format(dropout))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = 0.1, weight_decay = 0, activation_function='relu', momentum = 0, optimizer = 'SGD',
                                        dropout = dropout)
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')
            

f.close()

#%%

momentums = [0]

f = open("../Results/results_momentum.txt", "w")

for momentum in momentums:
    f.write("Training for {}.\n".format(momentum))
    parameters, epoch_res = main(path, method = 'DSSP', conversion_method = 'method_2', window_size = 17,
                                        batch_size = 50, hidden_size = 110, split = 0.2, shuffle = True, epochs = 50,  
                                        lr = 0.1, weight_decay = 0, activation_function='relu', momentum = momentum, optimizer = 'SGD',
                                        dropout = 0)
    f.write(parameters + '\n')
    for res in epoch_res:
        f.write(res + '\n')
        
    f.write('\n\n')

f.close()


