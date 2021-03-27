# rnn + bi-directional + multilayer
import torch
import torch.nn as nn


class Lstm_block(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first, bidirectional):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
        )
        
        self.linear = nn.Sequential(
            nn.ReLU(),   # activation
            nn.BatchNorm1d(self.hidden_size*2),   # normalize before linear layer
            nn.Linear(self.hidden_size*2, self.output_size),    # dim reduction
            nn.LogSoftmax(-1),    # get the probability value
        )
    
    def forward(self, x):
        # |x| = (bs, t, input_size)
        z, _ = self.lstm(x)  # output of nn.LSTM = output(for all t-step), (hidden_state, cell_state) 
        # |z| = (bs, t, hidden_size * # directions)
        z = z[:, -1, :].squeeze(1) # pick the last time-step output + squeeze   
        # |z| = (bs, hidden_size * #direction) = 2 dim
        y_pred = self.linear(z)
        # |y_pred| = (bs, 10)
        
        return y_pred
    
    
class Lstm_classifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first, bidirectional):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        super().__init__()
        
        self.classifier = Lstm_block(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.num_layers,
            self.batch_first,
            self.bidirectional
        )
    
    def forward(self, x):
        # |x| = (bs, t-step, input_size)
        y_pred = self.classifier(x)
        # |y_pred| = (bs, 10)
        
        return y_pred