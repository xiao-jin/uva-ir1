import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
from ranknet_model import RankNet
import argparse
import datetime
import torch
import torch.nn as nn
import time
import itertools
from early_stopping import EarlyStopping
import matplotlib.pyplot as plt

# CONSTANT PARAMS
OUTPUT_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 100

# HYPER PARAMS
SIGMA = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EVALUATION_FREQ = 500
DEFAULT_EPOCHS = 5


def train(data, lr, num_layers, patience, min_delta):
    early_stop = False
    speed_up = False
    ndcg_scores = []
    arr_scores = []
    steps = []
    step_counter = 0

    ranknet = RankNet(data.num_features, OUTPUT_SIZE, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(params=ranknet.parameters(),lr=lr)

    num_layers = 0
    for layer in list(ranknet.layers._modules.values())[0:-1]: # exclude the output layer
        num_layers += isinstance(layer, nn.Linear)

    model_name = 'ranknet_layers_%d_speedup_%s_lr_%f' % (num_layers, speed_up, lr)
    early_stopping = EarlyStopping(ranknet, model_name)

    for epoch in range(1, DEFAULT_EPOCHS+1):
        if early_stop:
            break

        start = time.time()
        batch_start = time.time()
        for qid in range(data.train.num_queries()):
            if early_stop:
                break
            train_loss = []
            ranknet.train()
            optimizer.zero_grad()

            docs = data.train.query_feat(qid)
            n = len(docs)

            if len(docs) >= 2:
                input = torch.tensor(docs).float().to(device)
                output = ranknet.forward(input).reshape(-1)

                if speed_up:
                    speed_up_gradient(output, data.train.query_labels(qid))
                else:
                    loss = pairwise_loss(output, data.train.query_labels(qid), SIGMA)
                    train_loss.append(loss.item() / (n * (n-1) / 2))
                    loss.backward()
                
                optimizer.step()

            if qid % PRINT_EVERY == 0:
                if speed_up:
                    print('EPOCH [%d] | BATCH [%d / %d] | %.2f seconds' % (epoch, qid, data.train.num_queries(), time.time() - batch_start))
                else:
                    print('TRAIN LOSS [%.2f] | EPOCH [%d] | BATCH [%d / %d] | %.2f seconds' % (np.mean(train_loss), epoch, qid, data.train.num_queries(), time.time() - batch_start))
                batch_start = time.time()
        
            if qid % DEFAULT_EVALUATION_FREQ == 0 :
                early_stop, results = test_model(ranknet, data, early_stopping, validation=True)
                ndcg_scores.append(results['ndcg'][0])
                arr_scores.append(results['relevant rank'][0])
                steps.append(step_counter)
            
            step_counter += 1

        print('Epoch %i took %f second' % (epoch, time.time() - start))
    test_model(ranknet, data, validation=False)
    plt.plot(steps, ndcg_scores, label='NDCG')
    plt.plot(steps, arr_scores, label='ARR')
    plt.legend()
    plt.title('RankNet scores')
    plt.show()


def speed_up_gradient(output, target):
    s1, s2 = split_output(output)
    S_ij = get_labels(target)
    lambda_i = SIGMA* (0.5 * (1 - S_ij) - 1 / (1 + torch.exp(SIGMA * (s1 - s2))))

    lambda_i.detach_()
    s1 *= lambda_i
    s1.sum().backward()


def split_output(output):
    pairs = torch.stack([output.repeat(output.shape[0],1).t().contiguous().view(-1), output.repeat(output.shape[0])],1).T
    return pairs[0], pairs[1]


def pairwise_loss(output, target, sigma):
    s1, s2 = split_output(output)
    S_ij = get_labels(target)
    sig = sigma * (s1 - s2)
    c = 0.5 * (1 - S_ij) * sig + torch.log(1 + torch.exp(-sig))
    return c.sum()


def get_labels(labels):
    """
    Get labels based on the documents relevance
    """
    labels = np.column_stack([labels.repeat(len(labels)), np.tile(labels, len(labels))]).T
    s1 = labels[0]
    s2 = labels[1]

    return torch.sign(torch.from_numpy(s1 - s2).to(device))


def test_model(ranknet, data, early_stopping=None, validation=False):
    """
    Test model
    Returns:    True to early stop
                False to keeps training
    """
    ranknet.eval()
    if (validation):
        dataset = data.validation
    else:
        dataset = data.test

    total_loss = 0
    validation_scores = torch.tensor([]).to(device)

    with torch.no_grad():
        for i in range(dataset.num_queries()):
            docs = dataset.query_feat(i)
            n = len(docs)

            input = torch.tensor(docs).float().to(device)
            output = ranknet.forward(input).reshape(-1)
            
            if n >= 2:
                loss = pairwise_loss(output, dataset.query_labels(i), SIGMA)
                total_loss += loss.item() / (n * (n - 1) / 2)

            validation_scores = torch.cat((validation_scores, output.clone().detach().view(-1)))            

        results = evl.evaluate(dataset, validation_scores.cpu() .numpy())
        print('ndcg:', results['ndcg'])
        total_loss = total_loss / dataset.num_queries()

        if (validation):
            print("\n", 'AVG VAL LOSS [{}]'.format(total_loss), "\n")
            return early_stopping.monitor(results), results
        else:
            print("\n", 'AVG TEST LOSS [{}]'.format(total_loss), "\n")
            return True, results

data = dataset.get_dataset().get_data_folds()[0]
data.read_data()

lrs = [1e-2, 1e-3, 1e-4]
hidden_layers = [2,3,4,5]

for lr in lrs:
    for layer in hidden_layers:
        train(data=data, lr=lr, num_layers=layer, patience=10, min_delta=0.001)