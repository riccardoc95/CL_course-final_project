import torch
import torch.nn as nn
import numpy as np

import torchvision
from torch.utils.data import Subset, TensorDataset, DataLoader

from random import shuffle
import matplotlib.pyplot as plt

from data import get_dataset 
from agent import Agent
from utils import *
from config import *


train_tasks, val_tasks = get_dataset(verbose=True)
agent = Agent(args, train_tasks, val_tasks, vanilla=False)

agent.validate()
random_model_acc = [i[0] for i in agent.acc.values()]
agent.reset_acc()
agent.train(verbose=True)

acc_at_end_arr = dict2array(agent.acc_end)
plot_accuracy_matrix(acc_at_end_arr)

acc_arr = dict2array(agent.acc)
plot_acc_over_time(acc_arr)

print(f"The average accuracy at the end of sequence is: {compute_average_accuracy(acc_at_end_arr):.3f}")
print(f"BWT:'{compute_backward_transfer(acc_at_end_arr):.3f}'")
print(f"FWT:'{compute_forward_transfer(acc_at_end_arr, random_model_acc):.3f}'")