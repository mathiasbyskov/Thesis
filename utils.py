
"""
Created on Wed Feb 19 22:50:39 2020

@author: Mathias Byskov Nielsen
"""


########################################################
#
#   Functions for parsing the sequence data
#
########################################################

def format_dataset(raw_file):
    """ 
	Function that modifies the raw data.

	Raw data can be downloaded from:
	http://www.compbio.dundee.ac.uk/jpred/about.shtml

	Input:
		raw_file: raw data file

	Output:
		Saves the formatted dataset in the same folder

	"""
    f = open('513_distribute.tar.gz', 'r')
    dataset = open('CB513_formatted.txt', 'w')
    
    for line in f:
        line = line.replace('\x00', '')
        
        if '513_distribute' in line:
            dataset.write('\n')
            start = line.index('513_distribute')
            stop = line.index('.all') + 4
            name = line[start:stop]
            dataset.write(name)
            
            dataset.write('\n')
            RES = line[line.index('RES:'):]
            dataset.write(RES)
            continue
        
        if ('DSSP' in line) or ('DSSPACC' in line) or ('STRIDE' in line) or ('RsNo' in line) or ('DEFINE' in line):
	        dataset.write(line)
            
        if line[0:5] == 'align':
	        dataset.write(line)
    
    f.close()
    dataset.close()
    return "Done"

def read_data(data, method):
    """
    Function that extracts protein-seqs and structures
    
    Input: 
        data: formatted dataset
        method: specifies which structure to use (DSSP, STRIDE or DEFINE)
        
    Output:
        proteins: List with protein-names
        seqs: List with raw sequences
        structures: List with 2nd structure
    """
    
    file = open(data, 'r')
    
    proteins = []
    seqs = []
    structures = []
    
    for line in file:
	    
	    # Extract protein
	    if '513_distribute' in line:
	        proteins.append(line.strip().split('/')[1].split('.al')[0])

	    # Extract sequence
	    if 'RES:' in line:
	        seqs.append(line[4:].strip().split(',')[:-1])
	        
	    if method == 'DSSP':
	        if ('DSSP:' in line) and (len(proteins) == (len(structures) + 1)):
	            structures.append(line[5:].strip().split(',')[:-1])
	    
	    if method == 'STRIDE':
	        if ('STRIDE' in line) and (len(proteins) == (len(structures) + 1)):
	            structures.append(line[7:].strip().split(',')[:-1])
	            
	    if method == 'DEFINE':
	        if ('DEFINE' in line) and (len(proteins) == (len(structures) + 1)):
	            structures.append(line[7:].strip().split(',')[:-1])
        
    file.close()
    
    formatted_structures = []
    corrected_dict = {'_': 'C', 'b': 'S'}
    
    for structure in structures:
    	modified_structure = []
    	for letter in structure:

    		if letter == '_' or letter == 'b':
    			modified_structure.append(corrected_dict[letter])
    		else:
    			modified_structure.append(letter)

    	formatted_structures.append(modified_structure)

    return proteins, seqs, formatted_structures

def clean_seqs(seqs, structures):
    """ Helper function: Removes X and ? from seqs and structures, respectively. """
    
    new_seqs = []
    new_structures = []
    
    # Loop over sequences
    for idx_1, seq in enumerate(seqs):
        indicies = []
        
        # Collect indicies to be removed
        for idx_2, letter in enumerate(seq):
            if letter == 'X':
                indicies.append(idx_2)
                
        for idx_2, letter in enumerate(structures[idx_1]):
            if letter == '?':
                indicies.append(idx_2)
                
        # Remove possible duplicates
        indicies = list(set(indicies)) 
        
		# Delete indicies from seqs and structures
        for index in sorted(indicies, reverse = True):
            del seq[index]
            del structures[idx_1][index]
            
        new_seqs.append(seq)
        new_structures.append(structures[idx_1])
        
    return new_seqs, new_structures

def eight_to_three_state(seqs, method):
    """
    Converts an 8-state-seq to a 3-state-seq
    
    """
    
    if method == 'method_1':
    	conversion_dict = {'H':'H', 'G':'H', 'E':'E', 'B':'E', 'S':'C', 'T':'C', 'I':'C', 'C':'C'}
    if method == 'method_2':
    	conversion_dict = {'H':'H', 'G':'C', 'E':'E', 'B':'C', 'S':'C', 'T':'C', 'I':'C', 'C':'C'}
    if method == 'method_3':
    	conversion_dict = {'H':'H', 'G':'H', 'E':'E', 'B':'E', 'S':'C', 'T':'C', 'I':'H', 'C':'C'}
    if method == 'method_4':
    	conversion_dict = {'H':'H', 'G':'H', 'E':'E', 'B':'C', 'S':'C', 'T':'C', 'I':'C', 'C':'C'}
    if method == 'method_5':
    	conversion_dict = {'H':'H', 'G':'H', 'E':'E', 'B':'C', 'S':'C', 'T':'C', 'I':'H', 'C':'C'}

    three_state_seqs = []

    for seq in seqs:
        three_state_seq =  []

        for letter in seq:
            three_state_seq.append(conversion_dict[letter])
    
        three_state_seqs.append(three_state_seq)

    return three_state_seqs

def onehot_encoder(seqs, vocab_dict):
    """
    Performs one-hot encoding to a sequence given a vocab_dict
    
    Input:
        seqs: List with sequences over alphabet A
        vocab_dict: Dictionary over alphabet A (converts letter to integers)
        
    Output:
        onehot_seqs: List with onehot-encodings for all sequences
    """
    
    onehot_seqs = [] 

    for seq in seqs:
        onehot_seq = []
        
        integer_encoded = [vocab_dict[char] for char in seq]
        
        for value in integer_encoded:
            onehot_vec = [0 for _ in range(len(vocab_dict.keys()))]
            onehot_vec[value - 1] = 1
            onehot_seq.append(onehot_vec)
        
        onehot_seqs.append(onehot_seq)

    return onehot_seqs

def latex_print(file):
    """ Extracts accuracies from results file and print them in latex-format. 

        Input:
            file: The result file produced in results_onelayer.py


    """
    
    f = open(file, 'r')
    
    experiment_dict = {}
    epoch = None
    num_epochs = "inf"
    
    for line in f:
        if 'Training' in line:
            parameter = line.strip().replace(',', '').replace('.', '').split(" ")
            parameter = parameter[-2] + ', ' + parameter[-1]

            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []        
    
        if '{' in line:
            num_epochs = int(line.replace(',', '').replace(':', '').replace('{', '').replace('}','').replace("'", "").split(' ')[19])
            
        if 'Epoch:' in line:
            res_list = line.strip().replace('-', '').replace('  ',' ').split(' ')
            
            epoch = int(res_list[1])
    
            train_loss.append('({},{})'.format(epoch, float(res_list[3])))
            train_acc.append('({},{})'.format(epoch, float(res_list[5])))
            test_loss.append('({},{})'.format(epoch, float(res_list[7])))
            test_acc.append('({},{})'.format(epoch, float(res_list[9])))
        
        if epoch == num_epochs:
            experiment_dict[parameter] = {'train_loss' : train_loss, 
                                          'train_acc'  : train_acc, 
                                          'test_loss'  : test_loss, 
                                          'test_acc'   : test_acc
                                          }
    
    
    f.close()
    
    return experiment_dict