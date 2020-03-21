import itertools
import torch.nn as nn
import torch

class RankNet(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(RankNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, 256))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers):
            self.layers.append(nn.Linear(256,256))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(256, output_size))
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input