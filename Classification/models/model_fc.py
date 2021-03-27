import torch
import torch.nn as nn

class Fc_block(nn.Module):
    
    def __init__(self, in_dim, out_dim, batch_norm):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        
        super().__init__()
        
        if self.batch_norm:
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
                nn.Dropout()
            )
        
    def forward(self, x):
        return self.block(x)

class Fc_classifier(nn.Module):
    
    def __init__(self, in_dim, out_dim, config):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b_norm = config.batch_norm
        
        super().__init__()
        
        self.layers = nn.Sequential(
            Fc_block(self.in_dim, 500, self.b_norm),
            Fc_block(500,400, self.b_norm),
            Fc_block(400,300, self.b_norm),
            Fc_block(300,200, self.b_norm),
            Fc_block(200,100, self.b_norm),
            nn.Linear(100,self.out_dim),
            nn.LogSoftmax(dim=-1)
        )
 
    def forward(self, x):
        # |x| = (bs, 784)
        y_pred = self.layers(x)
        # |y_pred| = (bs, 10)
        return y_pred