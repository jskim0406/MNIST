import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


class Trainer():
    
    def __init__(self, model, optim, crit, train_loader, valid_loader):
        
        self.model = model
        self.optim = optim
        self.crit = crit
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
       
    def _train(self, train_loader):
        
        train_loss = 0
        
        self.model.train()
        for x, y in train_loader:
            x, y = x.to('cuda:0'), y.to('cuda:0')
            y_hat = self.model(x)
            loss = self.crit(y_hat, y)
            
            self.optim.zero_grad()
            loss.backward()
            
            self.optim.step()
            
            train_loss += float(loss)
            
        train_loss /= len(train_loader)
        return train_loss
    
                          
    def _valid(self, valid_loader):
        
        valid_loss = 0
                          
        self.model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to('cuda:0'), y.to('cuda:0')
                y_hat = self.model(x)
                loss = self.crit(y_hat, y)

                valid_loss += float(loss)
                
            valid_loss /= len(valid_loader)
            return valid_loss
        
        
    def train(self, config):
        
        lowest_loss = np.inf
        best_model = None
        
        for epoch in range(config.n_epochs):
            
            train_loss = self._train(self.train_loader)
            valid_loss = self._valid(self.valid_loader)
            
            if lowest_loss > valid_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            
            if (epoch+1) % 5 == 0:
                print("epoch : %d, train_loss : %.4f, valid_loss : %.4f, lowest_loss : %.4f" % (epoch+1, 
                                                                                                train_loss,
                                                                                                valid_loss, 
                                                                                                lowest_loss,
                                                                                               ))
        # restore best model
        self.model.load_state_dict(best_model)
            
           