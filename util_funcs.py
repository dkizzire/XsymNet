import os
import time
import numpy as np
import pandas as pd
import json
import torch
from sklearn.utils.class_weight import compute_class_weight




def get_bravais_data_and_labels(y_train_sg_number_list, y_test_sg_number_list):
    with open('/home/dx6/ESS/dict_spacegroup_2_bravais_integer_labels.json', 'r') as f: 
        label_converter2 = json.load(f)

    y_train_bravais_init_str = [str(x) for x in y_train_sg_number_list]
    y_test_bravais_init_str = [str(x) for x in y_test_sg_number_list]
    y_train_bravais_init = [label_converter2.get(x) for x in y_train_bravais_init_str]
    y_test_bravais_init = [label_converter2.get(x) for x in y_test_bravais_init_str]

    # this block of code makes sure that target labels are in the range `t >= 0 && t < n_classes`
    classes=np.unique(y_train_bravais_init)
    int_labels_4_bravais = np.arange(len(classes))
    int_labels_4_bravais_map = {k: v for k,v in zip(classes,int_labels_4_bravais)}
    y_train_bravais = [int_labels_4_bravais_map.get(x) for x in y_train_bravais_init]
    y_test_bravais = [int_labels_4_bravais_map.get(x) for x in y_test_bravais_init]

    # create weights tensor to fix class imbalance
    class_weights_bravais = compute_class_weight('balanced', classes=np.unique(y_train_bravais), y=y_train_bravais)
    class_weights_bravais = torch.tensor(class_weights_bravais).float()
    y_train_bravais = torch.tensor(y_train_bravais)
    y_test_bravais = torch.tensor(y_test_bravais)
    
    return y_train_bravais, y_test_bravais, class_weights_bravais


def write_error_jsons(y_test_sg_number_list, incorrect, inv_map): 
    with open('/home/dx6/ESS/jacs_dict.json', 'r') as f: 
        jacs_info = json.load(f)

    incorrect_sg = []
    for i in range(len(incorrect)):
        temp_incorrect_sg = [inv_map.get(x) for x in incorrect[i]]
        incorrect_sg.append(temp_incorrect_sg)

    sg_true_negative = []
    for i in range(len(incorrect_sg)):
        for j in range(len(incorrect_sg[i])):
            if incorrect_sg[i][j] == None:
                continue
            else:
                sg_true_negative.append(incorrect_sg[i][j])
                break


    # create mapping dict with testing classes and class counts 
    y_test_list = [int(x) for x in y_test_sg_number_list]
    y_test_unique, y_test_unique_counts = np.unique(y_test_list, return_counts=True)
    y_test_map = {int(k):int(v) for k,v in zip(y_test_unique, y_test_unique_counts)}
    # create mapping dict with incorrectly labeled classes and how many of each
    unique_TN, TN_counts = np.unique(sg_true_negative, return_counts=True)
    unique_TN_list = [int(x) for x in unique_TN]
    # get the space group long and short name and origin of every incorrectly labeled class
    jacs_sg_names = [jacs_info.get(x) for x in unique_TN]
    # get the number of samples in the testing set for each incorrectly labeled class
    y_test_class_counts = [y_test_map.get(x) for x in unique_TN_list]
    percent_incorrect = [100*(x/y) for x,y in zip(TN_counts, y_test_class_counts)]
    # create a mapping dict with the space groups that were incorrectly labeled with: num_incorrect out of total, & the space group info
    TN_map = {int(k): [int(v), 'incorrect out of', a, b] for k, v, a, b in zip(unique_TN, TN_counts, y_test_class_counts, jacs_sg_names)}
    TN_map_percent = {int(k): [float(v), 'percent incorrect', b] for k, v, b in zip(unique_TN, percent_incorrect, jacs_sg_names)}
    # sort the dict by the most incorrectly labeled space group to the least (unfortunately a function of how many of each class were in the testing data set)
    sorted_TN_map = sorted(TN_map.items(), key=lambda x:x[1], reverse=True)
    converted_TN_map = dict(sorted_TN_map)
    json_object = json.dumps(TN_map, indent=4, sort_keys=True)
    json_object_sorted = json.dumps(converted_TN_map, indent=4)
    # sort the percentage incorrect dict by percentage incorrect
    sorted_TN_map_percent = sorted(TN_map_percent.items(), key=lambda x:x[1], reverse=True)
    converted_TN_map_percent = dict(sorted_TN_map_percent)
    json_object_sorted_percent = json.dumps(converted_TN_map_percent, indent=4)
    # write the sorted dicts to json files :: 1 sorted by space group in ascending order, 2 sorted by space group with the most incorrect classifications
    with open('/home/dx6/ESS/incorrect_predictions_by_SG.json', 'w') as f:
        f.write(json_object)
    with open('/home/dx6/ESS/incorrect_predictions_by_ERROR.json', 'w') as f:
        f.write(json_object_sorted)
    with open('/home/dx6/ESS/incorrect_predictions_by_PERCENT_ERROR.json', 'w') as f:
        f.write(json_object_sorted_percent)




# with open('/home/dx6/ESS/jacs_convert_sg_number_2_index.json', 'r') as f: 
#     label_converter = json.load(f)

# # convert labels (space group numbers) from storage tensor to list
# y_train_sg_number_list = y_train_sg_number.tolist()
# y_test_sg_number_list = y_test_sg_number.tolist()

# # convert space grounp numbers (ints) to strs
# y_train_sg_number_list = [str(x) for x in y_train_sg_number_list]
# y_test_sg_number_list = [str(x) for x in y_test_sg_number_list]

# # convert space group number (strs) to integer labels where integer labels (t) satisfy condition (`t >= 0 && t < n_classes`)
# y_train_init = [label_converter.get(x) for x in y_train_sg_number_list]
# y_test_init = [label_converter.get(x) for x in y_test_sg_number_list]