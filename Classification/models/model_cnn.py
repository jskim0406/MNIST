import torch
import torch.nn as nn

class Cnn_block(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # |x| = (bs, c_in, h_in, w_in)
        y = self.cnn_block(x)
        # |y| = (bs, c_out, h_out, w_out)
        return y
    
    
class Cnn_classifier(nn.Module):
    
    def __init__(self, out_dim):
        
        self.out_dim = out_dim
        
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            Cnn_block(1, 32),
            Cnn_block(32, 64),
            Cnn_block(64, 128),
            Cnn_block(128, 256),
            Cnn_block(256, 512),
            # |output| = (bs, 512, 1,1)
        )
        
        self.fc_layers = nn.Sequential(
            # |input| = (bs, 512)
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,out_dim),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        
        assert x.dim() >= 2
        
        if x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(-2), x.size(-1))
        
        z = self.conv_layers(x) # |z| = (bs, 512, 1, 1)
        z = z.squeeze() # |z| = (bs, 512)
        y = self.fc_layers(z) # |y| = (512, 10)
        
        return y
        
        
        
        
        
        