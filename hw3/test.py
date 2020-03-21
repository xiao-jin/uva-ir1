import dataset
import torch
import numpy as np

# data = dataset.get_dataset().get_data_folds()[0]
# data.read_data()

x = torch.tensor([1,2,3])
grid = torch.stack([x.repeat(x.shape[0]), x.repeat(x.shape[0],1).t().contiguous().view(-1)],1)

# grid = torch.stack([x.repeat(w), y.repeat(h,1).t().contiguous().view(-1)],1)
    

s = preds.repeat((len(preds),1)) - preds[:,None]
S = torch.sign(labels.repeat((len(labels),1)) - labels[:,None])

return torch.sum((gamma*((.5*(1-S)) - (1/(1+torch.exp(gamma*s))))).fill_diagonal_(0), axis = 1, keepdim = True)


pass