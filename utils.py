import numpy as np
import argparse
import copy
from mydataset.dataset import ScientificDataset
import yaml



def build_dataset(dargs, syn_length=False):
    all_datasets = []
    for name in dargs:
        all_datasets.append(ScientificDataset(dargs[name]))
            
    if syn_length:
        max_length = np.max([len(d) for d in all_datasets])
        
        for d in all_datasets:
            d.visble_length = max_length
            
    return all_datasets


def convert_args(args, train=True):
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    

    dnames = args.train_set.split(",") if train else args.test_set.split(",")
        
    all_args = {}
    
    
    for dname in dnames:
       
        
        cfg = config[dname]
        
        data_dict = {"name":dname, "data_path":cfg["data_path"]}
        
        if train:
            data_dict.update(cfg["train_subset"])
            data_dict.update(config["train_config"])
        else:
            data_dict.update(cfg["test_subset"])
            data_dict.update(config["test_config"])
            
        all_args[dname]=data_dict
        
    return all_args


def second_to_time(remaining_time):
    remaining_hours = int(remaining_time // 3600)
    remaining_minutes = int((remaining_time % 3600) // 60)
    remaining_seconds = int(remaining_time % 60)
    return remaining_hours, remaining_minutes, remaining_seconds