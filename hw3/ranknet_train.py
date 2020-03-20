import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
from ranknet_model import RankNet
import argparse
import datetime
import torch
import time
import itertools
import matplotlib.pyplot as plt

# CONSTANT PARAMS
OUTPUT_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPER PARAMS
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_EVALUATION_FREQ = 500
DEFAULT_EPOCHS = 5
SIGMA = 0.9
PRINT_EVERY = 100

def main():
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    ranknet = RankNet(data.num_features, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(params=ranknet.parameters(),lr=DEFAULT_LEARNING_RATE)

    for epoch in range(1, DEFAULT_EPOCHS+1):
        start = time.time()
        batch_start = time.time()
        for i in range(data.train.num_queries()):
            train_loss = []
            ranknet.train()
            optimizer.zero_grad()

            docs = data.train.query_feat(i)
            n = len(docs)
            input = torch.tensor(docs).float()
            output = ranknet.forward(input)

            if len(docs) >= 2:
                loss = pairwise_loss(output, data.train.query_labels(i), SIGMA)
                train_loss.append(loss.item() / (n * (n-1) / 2))
                loss.backward()
                optimizer.step()

            if i % PRINT_EVERY == 0:
                print('TRAIN LOSS [%.2f] | EPOCH [%d] | BATCH [%d / %d] | %.2f seconds' % (np.mean(train_loss), epoch, i, data.train.num_queries(), time.time() - batch_start))
                batch_start = time.time()
        
            if i > 0 and i % DEFAULT_EVALUATION_FREQ == 0 :
                test_model(ranknet, data, validation=True)

        print('Epoch %i took %f second' % (epoch, time.time() - start))

    test_model(ranknet, data, validation=False)


def pairwise_loss(output, target, sigma):
    S_ij = get_labels(target)

    n = output.shape[0]
    s1 = torch.zeros(int((n * (n - 1) / 2)))
    s2 = torch.zeros(int((n * (n - 1) / 2)))

    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            s1[counter] = output[i]
            s2[counter] = output[j]
            counter += 1
    
    sig = sigma * (s1 - s2)
    c = 0.5 * (1 - S_ij.view(-1, 1)) * sig + torch.log(1 + torch.exp(-sig))
    return c.sum()


def get_labels(labels):
    """
    Get labels based on the documents relevance
    """
    label_list = []
    ys = list(itertools.combinations(labels, 2))
    for i, j in ys:
        if i > j:
            label = 1
        elif i < j:
            label = -1
        elif i == j:
            label = 0
        label_list.append(label)

    return torch.tensor(label_list)


def test_model(ranknet, data, validation=False):
    ranknet.eval()
    if (validation):
        dataset = data.validation
    else:
        dataset = data.test

    total_loss = 0
    validation_scores = torch.tensor([])
    results_list = []

    with torch.no_grad():
        for i in range(dataset.num_queries()):
            docs = dataset.query_feat(i)
            n = len(docs)

            input = torch.tensor(docs).float()
            output = ranknet.forward(input)
            
            if n >= 2:
                loss = pairwise_loss(output, dataset.query_labels(i), SIGMA)
                total_loss += loss.item() / (n * (n - 1) / 2)

            validation_scores = torch.cat((validation_scores, output.clone().detach().view(-1)))
        

        results = evl.evaluate(data.validation, validation_scores.numpy(), print_results=True)
        results_list.append(results)

        total_loss = total_loss / dataset.num_queries()

        if (validation):
            print("\n", 'AVG VAL LOSS [{}]'.format(total_loss), "\n")
        else:
            print("\n", 'AVG TEST LOSS [{}]'.format(total_loss), "\n")


if __name__ == '__main__':
    main()
