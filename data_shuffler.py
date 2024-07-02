import os
import random
import numpy as np
import json


def combine_to_one(folder_path):
    '''
    Helper that combines all numpy files in a folder
    Returns tuple with (pf numpy array, jet numpy array)
    '''
    pfs = []
    jets = []
    for file in os.scandir(folder_path):

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
    np.savez(final_path + 'training_set.py', combine_combined(training_combined))
    np.savez(final_path + 'validation_set.py', combine_combined(validation_combined))
    np.savez(final_path + 'testing_set.py', combine_combined(testing_combined))
    


if __name__ == "__main__":
    # directory = '/path/to/directory'  # Replace with the path to your directory
    # ratios = {
    #     'training': 0.7,  # 70% of events for training
    #     'validation': 0.2,  # 20% of events for validation
    #     'testing': 0.1   # 10% of events for testing
    # }
    # main(directory, ratios)
    
    #testing:
    test_file = '/n/holystore01/LABS/iaifi_lab/Lab/monojet/GluGluHToBB'
    save_to_groups(test_file)

    # print(a.shape, b.shape, c.shape)

