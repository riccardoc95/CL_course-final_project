import torch

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")

args = {
    'lr': 1e-5, 
    'bs': 128, 
    'epochs': 50, 
    'num_tasks': 5,
    'dataset': "MNIST",
    'num_classes': 10, 
    'in_size': 28,
    'n_channels': 1,
    'hidden_size': 256,
    'samples_per_class':1000,
    'device': device
    }

gan_args = {
    'latent_dim': 1024,
    'input_height': 28,
    'input_width': 28,
    'input_dim': 64,
    'output_dim': 1,
    'batch_size': 64,
    'lr': 1e-4,
    'z_dim': 64,
    'epoch_number': 20,
    'nb_samples': 10,
    'samples_per_class':1000
}