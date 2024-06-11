import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
from gan import MemoryGAN
from config import *


class Agent:
    def __init__(self, args, train_datasets, val_datasets, vanilla=False):
        self.args = args
        self.vanilla = vanilla
        self.model = MLP(self.args)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.reset_acc()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        if not self.vanilla:
            self.memory_gan = MemoryGAN(
                latent_dim = gan_args['latent_dim'],
                input_height = gan_args['input_height'],
                input_width = gan_args['input_width'],
                input_dim = gan_args['input_dim'],
                output_dim = gan_args['output_dim'],
                batch_size = gan_args['batch_size'],
                lr = gan_args['lr'],
                z_dim = gan_args['z_dim'],
                epoch_number = gan_args['epoch_number'],
                nb_samples=gan_args['nb_samples'],
                device=device) 
    
    def reset_acc(self):
        self.acc = {key: [] for key in self.args['task_names']}
        self.acc_end = {key: [] for key in self.args['task_names']}


    def train(self, verbose=True):
        for task, data in self.train_datasets.items():
            if verbose:
                print(f"Task {task}")
            if self.vanilla:
                if int(task) == 0:
                    X_ = data[0]
                    y_ = data[1]
                #    continue
                else:
                    X_ = torch.cat((X_.to(self.args['device']), data[0].to(self.args['device'])), 0)
                    y_ = torch.cat((y_.to(self.args['device']), data[1].to(self.args['device'])), 0)
                #    if task != list(self.train_datasets.keys())[-1]:
                #        continue
            else:
                X_ = data[0]
                y_ = data[1]
                if int(task) > 0 :
                    self.model.eval()
                    with torch.no_grad():
                        X_replay = self.memory_gan.get_replay(self.args['samples_per_class'] * 2 * int(task))
                        y_replay = torch.argmax(self.model(X_replay),axis=1)

                        if verbose:
                            f, axarr = plt.subplots(1,10)
                            for j in range(10):
                                axarr[j].imshow(X_replay.cpu().numpy()[j, 0], cmap="gray")
                                axarr[j].set_title(f"{y_replay[j].cpu().numpy()}")
                                np.vectorize(lambda ax:ax.axis('off'))(axarr);
                            plt.show()

                        
                    X_ = torch.cat((X_.to(self.args['device']), X_replay.to(self.args['device'])), 0)
                    y_ = torch.cat((y_.to(self.args['device']), y_replay.to(self.args['device'])), 0)
                    


                self.model = MLP(self.args)
                self.model.to(device)
                self.model.train()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])

            data_ = TensorDataset(X_, y_)
            data_loader = DataLoader(data_, batch_size=self.args['bs'], shuffle=True)
     
            
            for epoch in range(self.args['epochs']):
                epoch_loss = 0
                total = 0
                correct = 0
                for e, (X, y) in enumerate(data_loader):
                    X, y = X.to(device), y.to(device)
                    output = self.model(X)
                    loss = self.criterion(output, y.long())
                    self.optimizer.zero_grad()
                    loss.backward() 
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    correct += torch.sum(torch.topk(output, axis=1, k=1)[1].squeeze(1) == y)
                    total += len(X)
                    if e % 50 == 0:
                        self.validate()
                    
                if verbose:
                    print(f"Epoch {epoch}: Loss {epoch_loss/(e+1):.3f} Acc: {correct/total:.3f}")
            self.validate(end_of_epoch=True)

            if not self.vanilla and task != list(self.train_datasets.keys())[-1]:
                # generator
                self.memory_gan.step(data[0], data[1])


    @torch.no_grad()
    def validate(self, end_of_epoch=False):
        self.model.eval()
        for task, data in self.val_datasets.items():
            X = data[0]
            y = data[1]
            data_ = TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(data_, batch_size=args['bs'], shuffle=True)
            correct, total = 0, 0
            for e, (X, y) in enumerate(loader):
                X, y = X.to(device), y.to(device)
                output = self.model(X)
                correct += torch.sum(torch.topk(output, axis=1, k=1)[1].squeeze(1) == y).item()
                total += len(X)
            self.acc[task].append(correct/total)
            if end_of_epoch:
                self.acc_end[task].append(correct/total)
        self.model.train()


class MLP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args['hidden_size']
        self.fc1 = torch.nn.Linear(args['in_size']**2 * args['n_channels'], hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, args['num_classes'])

    def forward(self, input):
        x = input.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x