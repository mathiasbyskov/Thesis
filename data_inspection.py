# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:05:25 2020

@author: Mathias
"""
#%%
import os 
os.chdir('C:/Users/mathi/Desktop/Thesis/Implementations')

from utils import read_data, clean_seqs, eight_to_three_state

path = '../DATA/CB513_formatted.txt'

proteins, seqs, structures = read_data(path, 'DSSP')
seqs, structures = clean_seqs(seqs, structures)

structures_3_state = eight_to_three_state(structures, method = 'method_1')
    


#### KEY FIGURES ####

# Number of seqs    
print("Number of seqs: {}".format(len(proteins)))

# Total number of aa's
length = 0
for seq in seqs:
    length += len(seq)
    
print("Total number os aa's: {}".format(length))

# Average length of seq

print("Average length of seq: {} \n".format(round(length / len(proteins), 2)))




#### Distribution of aa's
print("Distribution of aa's:\n")
aa_dict_count = {}

for seq in seqs:
    for aa in seq:
        if aa not in aa_dict_count:
            aa_dict_count[aa] = 1     
        else:
            aa_dict_count[aa] += 1

aa_dict_prop = {k: (v / length) * 100 for k, v in aa_dict_count.items()}

print('AA Distribution:', aa_dict_prop, '\n')


#%%
    
#### DSSP Distribution (8 states)
proteins, seqs, structures = read_data(path, 'DSSP')
seqs, structures = clean_seqs(seqs, structures)

length = 0
for seq in structures:
    length += len(seq)

eight_dict_count = {}

for structure in structures:
    for aa in structure:
        if aa not in eight_dict_count:
            eight_dict_count[aa] = 1     
        else:
            eight_dict_count[aa] += 1

eight_dict_prop = {k: (v / length) * 100 for k, v in eight_dict_count.items()}

print('DSSP distribution:', eight_dict_prop, '\n')

#### STRIDE Distribution (8 states)

proteins, seqs, structures = read_data(path, 'STRIDE')
seqs, structures = clean_seqs(seqs, structures)

eight_dict_count = {}

for structure in structures:
    for aa in structure:
        if aa not in eight_dict_count:
            eight_dict_count[aa] = 1     
        else:
            eight_dict_count[aa] += 1

eight_dict_prop = {k: (v / length) * 100 for k, v in eight_dict_count.items()}

print('STRIDE distribution:', eight_dict_prop, '\n')

#### DEFINE Distribution (8 states)

proteins, seqs, structures = read_data(path, 'DEFINE')
seqs, structures = clean_seqs(seqs, structures)

eight_dict_count = {}

for structure in structures:
    for aa in structure:
        if aa not in eight_dict_count:
            eight_dict_count[aa] = 1     
        else:
            eight_dict_count[aa] += 1

eight_dict_prop = {k: (v / length) * 100 for k, v in eight_dict_count.items()}

print('DEFINE distribution:', eight_dict_prop, '\n')

#%%


#### DSSP (3 states)
proteins, seqs, structures = read_data(path, 'DSSP')
seqs, structures = clean_seqs(seqs, structures)

structures = eight_to_three_state(structures, method = 'method_1')

three_dict_count = {}

for structure in structures:
    for aa in structure:
        if aa not in three_dict_count:
            three_dict_count[aa] = 1     
        else:
            three_dict_count[aa] += 1

three_dict_prop = {k: (v / length) * 100 for k, v in three_dict_count.items()}

print('DSSP 3-state distribution:', three_dict_prop, '\n')



