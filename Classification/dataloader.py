import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    def __init__(self, data, label, config, flatten):
        self.data = data
        self.label = label
        self.config = config
        
        super().__init__()
        
    def __getitem__(self, idx):
        x, y = self.data[idx], self.label[idx]
        
        if self.config.model == 'fc':
            x = x.view(-1)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    
def mnist_get_loaders(config, is_train):
    
    dataset = torchvision.datasets.MNIST('./data/MNIST', 
                                        train=is_train,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
    
    x, y = dataset.data, dataset.targets
    
    # scaling
    x = x / float(255)
    
    # flatten
    if config.model == 'fc':
        x = x.view(x.shape[0], -1) # |x| = (60000, 28*28)
    
    return x, y
    

def get_loaders(config):
    
    
    flatten = True if config.model == 'fc' else False
    
    # get train dataset
    x_train, y_train = mnist_get_loaders(config, is_train=True)
    
    # shuffling (train, valid)
    indices = torch.randperm(x_train.shape[0])
    x = torch.index_select(x_train, dim=0, index=indices)
    y = torch.index_select(y_train, dim=0, index=indices)
    
    # split (train, valid)
    train_cnts = int(x_train.shape[0]*config.train_ratio)
    valid_cnts = x_train.shape[0]-train_cnts
    cnts = [train_cnts, valid_cnts]
    
    x_train, x_valid = x_train.split(dim=0, split_size=cnts)
    y_train, y_valid = y_train.split(dim=0, split_size=cnts)
    
    # get test dataset
    x_test, y_test = mnist_get_loaders(config, is_train=False)
    
    
    train_loader = DataLoader(CustomDataset(x_train, y_train, config, flatten=flatten), 
                              batch_size=config.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(CustomDataset(x_valid, y_valid, config, flatten=flatten), 
                              batch_size=config.batch_size,
                              shuffle=False)
    test_loader = DataLoader(CustomDataset(x_test, y_test, config, flatten=flatten), 
                              batch_size=config.batch_size,
                              shuffle=False)
    
    return train_loader, valid_loader, test_loader
    
    
