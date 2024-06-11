import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from copy import deepcopy

from tqdm.auto import tqdm
from config import *


class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, 
                 latent_dim = 1024,
                 input_height = 28,
                 input_width = 28,
                 input_dim = 62,
                 output_dim = 1):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.maxPool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)
    
    def reinit(self):
      self.apply(self.weights_init)

    def forward(self, input):
        input = input.view(-1, self.input_dim)
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)



class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self,
                 input_height = 28,
                 input_width = 28,
                 input_dim = 1,
                 output_dim = 1,
                 latent_dim = 1024):
        super(Discriminator, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.latent_dim = latent_dim

        shape = 128 * (self.input_height // 4) * (self.input_width // 4)

        self.fc1_1 = nn.Linear(784, self.latent_dim)
        self.fc1_2 = nn.Linear(10, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim * 2, self.latent_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(self.latent_dim // 2)
        self.fc3 = nn.Linear(self.latent_dim // 2, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(shape, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Sigmoid(),
        )
        self.aux_linear = nn.Linear(shape, 10)
        self.softmax = nn.Softmax()
        self.apply(self.weights_init)
    
    def reinit(self):
      self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.data.shape[0], 128 * (self.input_height // 4) * (self.input_width // 4))
        return self.fc(x)


class MemoryGAN:
    def __init__(self,
                 latent_dim = 1024,
                 input_height = 28,
                 input_width = 28,
                 input_dim = 62,
                 output_dim = 1,
                 batch_size = 64,
                 lr = 0.001,
                 z_dim = 62,
                 epoch_number = 20,
                 nb_samples=10,
                 device="cpu"):    

        self.batch_size = batch_size
        self.lr = lr
        self.z_dim = z_dim
        self.epoch_number = epoch_number
        self.nb_samples = nb_samples

        self.device = device
        
        self.G = Generator(latent_dim,
                           input_height,
                           input_width,
                           input_dim,
                           output_dim)
        self.D = Discriminator(input_height,
                               input_width,
                               1,
                               output_dim,
                               latent_dim)
        self.G.train()
        self.D.train()
        self.G.to(self.device)
        self.D.to(self.device)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)
        self.BCELoss = nn.BCELoss()

        self.set_generator()
        self.total_tasks = 0

    def set_generator(self):
        self.generator = deepcopy(self.G)
        
    def run_batch(self, x):
        x = x.view((-1, 1, 28, 28)) 
        # y_real and y_fake are the label for fake and true data
        y_real = Variable(torch.ones(x.size(0), 1))
        y_fake = Variable(torch.zeros(x.size(0), 1))
        
        y_real, y_fake = y_real.to(self.device), y_fake.to(self.device)
        
        z = torch.rand((x.size(0), self.z_dim))
        x, z = Variable(x), Variable(z)
        x, z = x.to(self.device), z.to(self.device)
        
        # update D network
        self.D_optimizer.zero_grad()
        D_real = self.D(x)
        D_real_loss = self.BCELoss(D_real, y_real[:x.size(0)])
        
        G = self.G(z)
        D_fake = self.D(G)
        D_fake_loss = self.BCELoss(D_fake, y_fake[:x.size(0)])
        
        D_loss = D_real_loss + D_fake_loss
        
        D_loss.backward()
        self.D_optimizer.step()
        self.G_optimizer.zero_grad()
        
        G = self.G(z)
        D_fake = self.D(G)
        G_loss = self.BCELoss(D_fake, y_real[:x.size(0)])
        
        G_loss.backward()
        self.G_optimizer.step()

    # function to get digits from only one class
    def get_iter_dataset(self, x_train, t_train, classe=None):
        if classe is not None:
            return x_train[torch.where(t_train==classe)[0]]

    # New function to generate samples for replay
    def get_replay(self, size):
        z = Variable(torch.rand((size, self.z_dim)))
        z = z.to(self.device)
        return self.generator(z)#.cpu()


    def step(self, x_train, t_train):      
        for task in tqdm(torch.unique(t_train), leave=False):
            #print(task)
            data = self.get_iter_dataset(x_train, t_train, classe=task)
            nb_batch = int(len(data)/ self.batch_size)
            
            for epoch in tqdm(range(self.epoch_number), leave=False):
                for index in range(nb_batch):
                    x = data[index*self.batch_size:(index+1)*self.batch_size]
                
                    if int(self.total_tasks) > 0 :
                        # We concat a batch of previously learned data
                        # the more there is past task more data needs to be regenerate
                        replay = self.get_replay(gan_args['batch_size'] * self.total_tasks)
                        x = torch.cat((x.to(self.device), replay.to(self.device)), 0)
                  
                    self.run_batch(x.to(self.device))

            self.total_tasks += 1
                  
            self.set_generator()
            
    def plot(self):
        z = Variable(torch.rand((self.nb_samples, self.z_dim)))
        
        z = z.to(self.device)
        samples = self.G(z).data
        samples = samples.cpu().numpy()
        
        f, axarr = plt.subplots(1,self.nb_samples)
        for j in range(self.nb_samples):
            axarr[j].imshow(samples[j, 0], cmap="gray")
            np.vectorize(lambda ax:ax.axis('off'))(axarr);


if __name__ == "__main__":
    device = "mps"
    
    import mnist
    mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()

    # function to get digits from only one class
    def get_iter_dataset(x_train, t_train, classe=None):
        if classe is not None:
            return x_train[np.where(t_train==classe)[0]]


    mg = MemoryGAN(
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


    classe=2
    x_data2 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data2 = torch.ones(x_data2.shape[0]) * classe
    
    classe=3
    x_data3 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data3 = torch.ones(x_data3.shape[0]) * classe
    
    x_data = torch.cat([x_data2,x_data3])
    t_data = torch.cat([t_data2,t_data3])
    mg.step(x_data.to(device), t_data.to(device))

    mg.plot()
    plt.show()
    

    classe=0
    x_data0 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data0 = torch.ones(x_data0.shape[0]) * classe
    
    classe=1
    x_data1 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data1 = torch.ones(x_data1.shape[0]) * classe
    
    x_data = torch.cat([x_data0,x_data1])
    t_data = torch.cat([t_data0,t_data1])
    mg.step(x_data.to(device), t_data.to(device))

    mg.plot()
    plt.show()
    
    classe=4
    x_data4 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data4 = torch.ones(x_data4.shape[0]) * classe
    
    classe=5
    x_data5 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data5 = torch.ones(x_data5.shape[0]) * classe
    
    x_data = torch.cat([x_data4,x_data5])
    t_data = torch.cat([t_data4,t_data5])
    mg.step(x_data.to(device), t_data.to(device))

    mg.plot()
    plt.show()
    
    classe=6
    x_data6 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data6 = torch.ones(x_data6.shape[0]) * classe
    
    classe=7
    x_data7 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data7 = torch.ones(x_data7.shape[0]) * classe
    
    x_data = torch.cat([x_data6,x_data7])
    t_data = torch.cat([t_data6,t_data7])
    mg.step(x_data.to(device), t_data.to(device))

    mg.plot()
    plt.show()
    
    classe=8
    x_data8 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data8 = torch.ones(x_data8.shape[0]) * classe
    
    classe=9
    x_data9 = torch.from_numpy(get_iter_dataset(x_train, t_train, classe=classe))
    t_data9 = torch.ones(x_data9.shape[0]) * classe
    
    x_data = torch.cat([x_data8,x_data9])
    t_data = torch.cat([t_data8,t_data9])
    mg.step(x_data.to(device), t_data.to(device))

    mg.plot()
    plt.show()