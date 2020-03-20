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
PRINT_EVERY = 100

# HYPER PARAMS
SIGMA = 1
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EVALUATION_FREQ = 500
DEFAULT_EPOCHS = 5

speed_up = True

def main():
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    ranknet = RankNet(data.num_features, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(params=ranknet.parameters(),lr=DEFAULT_LEARNING_RATE)

    for epoch in range(1, DEFAULT_EPOCHS+1):
        start = time.time()
        batch_start = time.time()
        for qid in range(data.train.num_queries()):
            train_loss = []
            ranknet.train()
            optimizer.zero_grad()

            docs = data.train.query_feat(qid)
            n = len(docs)

            if len(docs) >= 2:
                input = torch.tensor(docs).float()
                output = ranknet.forward(input)

                if speed_up:
                    speed_up_gradient(output, data.train.query_labels(qid))
                else:
                    loss = pairwise_loss(output, data.train.query_labels(qid), SIGMA)
                    train_loss.append(loss.item() / (n * (n-1) / 2))
                    loss.backward()
                
                optimizer.step()

            if qid % PRINT_EVERY == 0:
                print('TRAIN LOSS [%.2f] | EPOCH [%d] | BATCH [%d / %d] | %.2f seconds' % (np.mean(train_loss), epoch, qid, data.train.num_queries(), time.time() - batch_start))
                batch_start = time.time()
        
            if qid > 0 and qid % DEFAULT_EVALUATION_FREQ == 0 :
                test_model(ranknet, data, validation=True)

        print('Epoch %i took %f second' % (epoch, time.time() - start))
    test_model(ranknet, data, validation=False)


def speed_up_gradient(output, target):
    s1, s2 = split_output(output)
    S_ij = get_labels(target)
    lambda_i = torch.zeros_like(s1)
    for i in range(len(s1)):
        lambda_i[i] = SIGMA* (0.5 * (1 - S_ij[i]) - 1 / (1 + torch.exp(SIGMA * (s1[i] - s2[i]))))

    lambda_i.detach_()
    s1 *= lambda_i
    s1.sum().backward()


def split_output(output):
    n = output.shape[0]
    s1 = torch.zeros(int((n * (n - 1) / 2)))
    s2 = torch.zeros(int((n * (n - 1) / 2)))

    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            s1[counter] = output[i]
            s2[counter] = output[j]
            counter += 1

    return s1, s2


def pairwise_loss(output, target, sigma):
    s1, s2 = split_output(output)
    S_ij = get_labels(target)
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
    ndcg_list = []

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

        results = evl.evaluate(data.validation, validation_scores.numpy())
        print('ndcg:', results['ndcg'])
        ndcg_list.append(results['ndcg'])

        total_loss = total_loss / dataset.num_queries()

        if (validation):
            print("\n", 'AVG VAL LOSS [{}]'.format(total_loss), "\n")
        else:
            print("\n", 'AVG TEST LOSS [{}]'.format(total_loss), "\n")


if __name__ == '__main__':
    main()
