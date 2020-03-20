import itertools
import torch.nn as nn
import torch

class RankNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RankNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2024),
            nn.ReLU(),
            nn.Linear(2024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        
    def forward(self, input):
        return self.layers(input)