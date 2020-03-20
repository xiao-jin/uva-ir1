import itertools
import torch.nn as nn
import torch

class RankNet(nn.Module):
    def __init__(self, input_size, output_size, neg_slope):
        super(RankNet, self).__init__()

        self.sigma = torch.nn.Parameter(torch.tensor(1.))
        # self.layers = nn.Sequential(
        #     nn.Linear(input_size, 128, bias=True),
        #     nn.LeakyReLU(neg_slope),
        #     nn.BatchNorm1d(128),

        #     nn.Linear(input_size, 512, bias=True),
        #     nn.LeakyReLU(neg_slope),
        #     nn.BatchNorm1d(512),

        #     nn.Linear(512, 64, bias=True),
        #     nn.LeakyReLU(neg_slope),
        #     nn.BatchNorm1d(64),
            
        #     nn.Linear(64, output_size),
        # )
        
        self.layers = nn.ModuleList()


        self.layers.append(nn.Linear(input_size, 128))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(128, 64))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(64, 32))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(32, output_size))        

    def forward(self, x1, x2):
        """
        Gets a vector of documents
        Create all combinations
        """
        input = torch.cat((x1, x2))

        # s1 = self.step(x1)
        # s2 = self.step(x2)

        output = self.step(input)
        s1, s2 = torch.chunk(output, 2, dim=0)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        if torch.isnan(output).sum().item():
            print(output)
            raise Exception('Got NaN')

        return s1, s2

    def step(self, input):
        for layer in self.layers:
            input = layer.forward(input)
            if torch.isnan(input).sum().item():
                print(input)
        
        return input

    def weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)


    def loss(self, S_ij, s1, s2):
        sig = self.sigma * (s1 - s2)
        c = 0.5 * (1 - S_ij.view(-1, 1)) * sig + torch.log(1 + torch.exp(-sig))

        if torch.isnan(c).sum().item():
            print(c.sum())
        return c.sum()