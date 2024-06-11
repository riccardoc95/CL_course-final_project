class GreedyBuffer:
    def __init__(self, samples_per_class):
        self.samples_per_class = samples_per_class
        self.samples = torch.Tensor([])
        self.targets = torch.Tensor([])

    def store_data(self, loader):
        samples, targets = torch.Tensor([]), torch.Tensor([])
        for sample, target in loader:
            samples = torch.cat((samples, sample))
            targets = torch.cat((targets, target))
        
        for label in torch.unique(targets):
            greedy_idx = torch.where(targets == label)[0][:self.samples_per_class]
            self.samples = torch.cat((self.samples, samples[greedy_idx]))
            self.targets = torch.cat((self.targets, targets[greedy_idx]))

    def get_data(self):
        return self.samples, self.targets.to(torch.int64)

    def __len__(self):
        assert len(self.samples) == len(self.targets), f"Incosistent lengths of data tensor: {self.samples.shape}, target tensor: {self.targets.shape}!"
        return len(self.samples)
