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

# CONSTANT PARAMS
OUTPUT_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPER PARAMS
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_EVALUATION_FREQ = 500
DEFAULT_EPOCHS = 5
DEFAULT_NEG_SLOP = 0.02
PRINT_EVERY = 50

def main():
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    # batch the data, skipping last batch
    X_batched = [data.train.feature_matrix[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.train.feature_matrix), DEFAULT_BATCH_SIZE)][0:-1]
    y_batched = [data.train.label_vector[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.train.label_vector), DEFAULT_BATCH_SIZE)][0:-1]

    ranknet = RankNet(data.num_features, OUTPUT_SIZE, DEFAULT_NEG_SLOP).to(device)
    optimizer = torch.optim.Adam(params=ranknet.parameters(),lr=DEFAULT_LEARNING_RATE)
    train_loss = []

    for epoch in range(1, DEFAULT_EPOCHS+1):
        start = time.time()
        for i, batch in enumerate(X_batched):
        # for i in range(len(data.train.feature_matrix)):
        #     # xj = [data.train.feature_matrix[j] for j in range(i + 1, len(data.train.feature_matrix))]
        #     for j in range(i + 1, len(data.train.feature_matrix)):
            ranknet.train()
            batch_start = time.time()
            optimizer.zero_grad()

            # xi = torch.tensor(data.train.feature_matrix[i]).float().to(device)
            # xj = torch.tensor(data.train.feature_matrix[j]).float().to(device)
            s1, s2 = ranknet.forward(batch)

            labels = get_labels(y_batched[i])

            loss = ranknet.loss(labels, s1, s2)

            if torch.isnan(loss):
                continue

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            
            if i % PRINT_EVERY == 0:
                print('TRAIN LOSS [{}] | EPOCH [{}] | BATCH [{} / {}] | {} seconds'.format(loss.item(), epoch, i, len(X_batched), time.time() - batch_start))
                # print('TRAIN LOSS [{}] | EPOCH [{}] | i = {} j = {} | {} seconds'.format(np.mean(train_loss), epoch, i, j, len(X_batched), time.time() - batch_start))


        test_model(ranknet, data, validation=True)

        print('Epoch %i took %f second' % (epoch, time.time() - start))

    test_model(ranknet, data, validation=False)


def get_labels(labels):
    label_list = []
    ys = list(itertools.combinations(labels, 2))
    for label_i, label_j in ys:
        if label_i > label_j:
            label = 1
        elif label_i < label_j:
            label = -1
        elif label_i == label_j:
            label = 0
        label_list.append(label)

    return torch.tensor(label_list)


def test_model(ranknet, data, validation=False):
    ranknet.eval()
    if (validation):
        X_batched = [data.validation.feature_matrix[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.validation.feature_matrix), DEFAULT_BATCH_SIZE)][0:-1]
        y_batched = [data.validation.label_vector[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.validation.label_vector), DEFAULT_BATCH_SIZE)][0:-1]
    else:
        X_batched = [data.test.feature_matrix[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.test.feature_matrix), DEFAULT_BATCH_SIZE)][0:-1]
        y_batched = [data.test.label_vector[i:i+DEFAULT_BATCH_SIZE] for i in range(0, len(data.test.label_vector), DEFAULT_BATCH_SIZE)][0:-1]

    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(X_batched):
            s1, s2 = ranknet.forward(batch)

            labels = get_labels(y_batched[i])

            loss = ranknet.loss(labels, s1, s2)
            total_loss += loss.item()

        total_loss = total_loss / len(X_batched)

        if (validation):
            print("\n", 'AVG VAL LOSS [{}]'.format(total_loss), "\n")
        else:
            print("\n", 'AVG TEST LOSS [{}]'.format(total_loss), "\n")

        return


if __name__ == '__main__':
    main()
