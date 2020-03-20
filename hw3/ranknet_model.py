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
        
        self.layers.append(nn.Linear(128, 512))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(512, 64))
        self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(64, output_size))
        
        

    def forward(self, x):
        """
        Gets a vector of documents
        Create all combinations
        """
        comb = list(itertools.combinations(x, 2))
        # x1 = torch.tensor([item1 for (item1, item2) in comb]).float()
        # x2 = torch.tensor([item2 for (item1, item2) in comb]).float()

        # x1, x2 = torch.tensor(list(zip(*comb))).float()

        input = torch.tensor(comb).float()

        # s1 = self.layers(x1)
        # s2 = self.layers(x2)

        # s1 = self.step(x1)
        # s2 = self.step(x2)

        output = self.step(input)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        if torch.isnan(output).sum().item():
            print(output)

        return output

    def step(self, input):
        for layer in self.layers:
            input = layer.forward(input)
            if torch.isnan(input).sum().item():
                print(input)
        
        return input

    def loss2(self, labels, input):
        s1 = input[:,0,:]
        s2 = input[:,1,:]
        
        sig = self.sigma * (s1 - s2)
        c = 0.5 * (1 - labels.view(-1, 1)) * sig + torch.log(1 + torch.exp(-sig))

        if torch.isnan(c).sum().item():
            print(c.sum())
        return c.sum()


    def loss(self, labels, s1, s2):
        s = torch.zeros_like(s1)

        for i in range(len(s1)):
            if s1[i] > s2[i]:
                s[i] = 1
            elif s1[i] < s2[i]:
                s[i] = -1
            elif s1[i] == s2[i]:
                s[i] = 0
        sig = self.sigma * (s1 - s2)
        c = 0.5 * (1 - labels) * sig + torch.log(1 + torch.exp(-sig))

        if torch.isnan(c).sum().item():
            print(c.sum())
        return c.sum()