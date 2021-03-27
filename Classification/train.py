import torch
import torch.nn as nn
import torch.optim as optim

import argparse

from dataloader import get_loaders
from models.model_fc import Fc_classifier
from models.model_cnn import Cnn_classifier
from models.model_lstm import Lstm_classifier

from trainer import Trainer


def Argparse():
    
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required = True)
    p.add_argument('--model_fn', type=str, required = True)
    p.add_argument('--gpu_id', type=int, default = 0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--optim', type=str, default='adam')
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--n_epochs', type=int, default=30)
    p.add_argument('--batch_norm', type=str, default=True)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)

    
    config = p.parse_args()
    return config
    
    
def main(config):
    
    '''
    1. set device
    2. get data_loaders
    3. set model
    4. set optim, criterion
    5. set trainer
    6. train!
    7. CLI command
    '''
    
    # set device
    device = torch.device("cpu") if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    
    # prepare dataloader
    train_loader, valid_loader, test_loader = get_loaders(config)

    # set model
    if config.model == 'fc':
        model = Fc_classifier(28*28, 10, config).to(device)
    elif config.model == 'cnn':
        model = Cnn_classifier(10).to(device)
    elif config.model == 'lstm':
        model = Lstm_classifier(28, config.hidden_size, 10, config.num_layers, batch_first=True, bidirectional=True).to(device)

    # set hyper parameter, optim, crit'
    import torch.optim as optim
    
    if config.optim == 'adam':
        optim = optim.Adam(model.parameters())
    elif config.optim == 'sgd':
        optim = optim.SGD(model.parameters(), lr=config.lr)
    
    crit = nn.NLLLoss()
    
    print(model)
    
    # set trainer
    trainer = Trainer(model, optim, crit, train_loader, valid_loader)
    
    # train!
    trainer.train(config)
    
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optim.state_dict(),
        'config': config,
    }, config.model_fn)
 
# CLI command
if __name__ == '__main__':
    config = Argparse()
    main(config)
    
    