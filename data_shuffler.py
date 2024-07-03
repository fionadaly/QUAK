import os
import random
import numpy as np
import json

def split_list(input_list, number_of_sublists):
    '''
    Helper function from ChatGPT
    '''
    # Calculate the total length of the list
    total_length = len(input_list)
    
    # Calculate the base size of each sublist
    base_size = total_length // number_of_sublists
    
    # Calculate the number of lists that will have one extra element
    remainder = total_length % number_of_sublists
    
    # Initialize the result list with empty lists
    result = [[] for _ in range(number_of_sublists)]
    
    # Initialize the starting index
    start_index = 0
    
    for i in range(number_of_sublists):
        # Determine the size of the current sublist
        current_size = base_size + (1 if i < remainder else 0)
        
        # Assign the slice of the input list to the current sublist
        result[i] = input_list[start_index:start_index + current_size]
        
        # Update the starting index
        start_index += current_size
    
    return result

def combine_to_one(folder_path):
    '''
    Helper that combines all numpy files in a folder
    Returns tuple with (pf numpy array, jet numpy array)
    '''
    pfs = []
    jets = []
    for folder in folder_path:
        for file in os.scandir(folder):
            data = np.load(file)
            pf, jet = data['pf'], data['jet']
            pfs.append(pf)
            jets.append(jet)
    pf_array = np.concatenate(pfs)
    jet_array = np.concatenate(jets)
    return (pf_array, jet_array)


def save_to_groups(folder_path, final_path = None, training_ratio = 0.8, testing_ratio = 0.1, validation_ratio = 0.1):
    '''
    Takes in the name of a folder containing numpy files
    '''
    assert training_ratio + testing_ratio + validation_ratio == 1, print("ratios must add to 1!")
    pfs, jets = combine_to_one(folder_path)

    #default final_path
    if final_path == None:
        final_path = folder_path

    #combined is a zipped list with tuples (pf, jet) for each event
    combined = list(zip(pfs, jets)) 
    #randomizing the order
    random.shuffle(combined)
    
    total_events = len(combined)
    training_end = int(total_events*training_ratio)
    validation_end = training_end + int(total_events*validation_ratio)

    training_combined = combined[:training_end]
    training_split = split_list(training_combined, 5)
    validation_combined = combined[training_end:validation_end]
    testing_combined = combined[validation_end:]
    
    
    def combine_combined(combined):
        #Turns zipped list back into {'pf': full pf array, 'jet': full jet array}
        pf_list = []
        jet_list = []
        for (pf, jet) in combined:
            pf_list.append(pf)
            jet_list.append(jet)
        # print(len(pf_list), len(jet_list))
        pfs = np.vstack(pf_list)
        jets = np.vstack(jet_list)
        return {'pf': pfs, 'jet': jets}
        
    #saving data
    for i in len(training_split):
        np.savez_compressed(final_path + 'training_set' + str(i) + '.py', combine_combined(training_split[i]))
    np.savez_compressed(final_path + 'validation_set.py', combine_combined(validation_combined))
    np.savez_compressed(final_path + 'testing_set.py', combine_combined(testing_combined))
    

def split_data(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=10000, output_path="output"):
    '''
    File_path should be a LIST of file names
    '''
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."
    
    # Create output directories if they don't exist
    train_output_path = os.path.join(output_path, 'training')
    val_output_path = os.path.join(output_path, 'validation')
    test_output_path = os.path.join(output_path, 'testing')
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)
    
    # Counters for file naming
    train_count = val_count = test_count = 0
    
    def save_batch(data, count, dataset_type):
        pf_list, jet_list = zip(*data)
        if dataset_type == 'train':
            np.savez_compressed(os.path.join(train_output_path, f"train_set_{count}.npz"), pf=np.vstack(pf_list), jet=np.vstack(jet_list))
        elif dataset_type == 'val':
            np.savez_compressed(os.path.join(val_output_path, f"val_set_{count}.npz"), pf=np.vstack(pf_list), jet=np.vstack(jet_list))
        elif dataset_type == 'test':
            np.savez_compressed(os.path.join(test_output_path, f"test_set_{count}.npz"), pf=np.vstack(pf_list), jet=np.vstack(jet_list))
    
    def process_file(file):
        nonlocal train_count, val_count, test_count
        data = np.load(file)
        pf_data, jet_data = data['pf'], data['jet']
        
        combined_data = list(zip(pf_data, jet_data))
        random.shuffle(combined_data)
        
        train_data, val_data, test_data = [], [], []
        
        for pf, jet in combined_data:
            rand_val = random.random()
            if rand_val < train_ratio:
                train_data.append((pf, jet))
            elif rand_val < train_ratio + val_ratio:
                val_data.append((pf, jet))
            else:
                test_data.append((pf, jet))
            
            if len(train_data) >= batch_size:
                save_batch(train_data, train_count, 'train')
                train_data = []
                train_count += 1
            
            if len(val_data) >= batch_size:
                save_batch(val_data, val_count, 'val')
                val_data = []
                val_count += 1
            
            if len(test_data) >= batch_size:
                save_batch(test_data, test_count, 'test')
                test_data = []
                test_count += 1
        
        # Save any remaining data
        if train_data:
            save_batch(train_data, train_count, 'train')
            train_count += 1
        if val_data:
            save_batch(val_data, val_count, 'val')
            val_count += 1
        if test_data:
            save_batch(test_data, test_count, 'test')
            test_count += 1
    
    # Process each file in the directory
    for folder in file_path:
        for file in os.scandir(folder):
            if file.name.endswith('.npz'):
                process_file(file.path)

def monojet_specific_file_iterator(file_list):
    '''
    Specific to the folder that I take data from
    '''
    return ['/n/holystore01/LABS/iaifi_lab/Lab/monojet/' + i for i in file_list]


if __name__ == "__main__":
    #create a list of files you would like to iterate over
    '''
    GluGluHToBB       QCD_HT1500to2000  QCD_HT700to1000  TTToHadronic      ZJetsToQQ_HT-600to800
    QCD_HT1000to1500  QCD_HT2000toInf   TTTo2L2Nu        TTToSemiLeptonic  ZJetsToQQ_HT-800toInf
    '''
    qcd_addresses = ['QCD_HT1500to2000', 'QCD_HT700to1000', 'QCD_HT1000to1500', 'QCD_HT2000toInf']
    qcd_addresses = monojet_specific_file_iterator(qcd_addresses)

    z_addresses = ['ZJetsToQQ_HT-600to800', 'ZJetsToQQ_HT-800toInf']
    z_addresses = monojet_specific_file_iterator(z_addresses)

    #run the split data fuction, the output path should ALREADY exist in your directory

    split_data(qcd_addresses, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=50000, output_path='Data/QCD_HT')



