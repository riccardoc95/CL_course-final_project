import torch
import numpy as np
import mnist
from random import shuffle
from config import *


mnist.init()
x_train, t_train, x_test, t_test = mnist.load()

def get_dataset(verbose=True):
    classes = list(range(args['num_classes']))
    shuffle(classes)
    class_split = {str(i): classes[i*2: (i+1)*2] for i in range(args['num_tasks'])}
    args['task_names'] = list(class_split.keys())
    if verbose:
        print("class_split:", class_split)
    
    train_tasks = split_dataset(x_train, t_train, class_split, device=args['device'])
    val_tasks = split_dataset(x_test, t_test, class_split, device=args['device'])
    return train_tasks, val_tasks

# function to get digits from only one class
def get_iter_dataset(x_train, t_train, classe=None):
    if classe is not None:
        return x_train[np.where(t_train==classe)[0]]

def split_dataset(x, t, tasks_split, device=args['device']):
    split_dataset = {}
    for e, current_classes in tasks_split.items():
        for i, current_class in enumerate(current_classes):
            x_data_current = torch.from_numpy(get_iter_dataset(x, t, classe=current_class))
            t_data_current = torch.ones(x_data_current.shape[0]) * int(current_class)
            if i == 0:
                x_data = x_data_current
                t_data = t_data_current
            else:
                x_data = torch.cat([x_data,x_data_current])
                t_data = torch.cat([t_data,t_data_current])
        
        split_dataset[e] = (x_data, t_data)
        #print(e, torch.unique(t_data))
    return split_dataset