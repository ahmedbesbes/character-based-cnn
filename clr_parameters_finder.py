'''
This script allows to find the optimal parameters for a learning rate scheduling:

- min_lr 
- max_lr 

We vary the learning rate inside one or several epochs between start_lr and end_lr 
(given as arguments) and for each mini-batch, we note the value of the learning 
rate and the loss.

Then we plot the loss versus the learning rate and save it to plots/

There's in general a downward trend first, a minimum and upward trend.

One heuristic to find the optimal parameters:

- max_lr = argmin_lr(loss) / 10
- mix_lr = max_lr / 10


reference: https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

'''

import math
import os
import shutil
import json
import argparse
import time
from datetime import datetime
from collections import Counter

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from src.data_loader import MyDataset, load_data
from src import utils
from src.model import CharacterLevelCNN

from matplotlib import pyplot as plt


def run(args):

    batch_size = args.batch_size

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": args.workers}

    texts, labels, number_of_classes, sample_weights = load_data(args)
    train_texts, _, train_labels, _, _, _ = train_test_split(texts,
                                                             labels,
                                                             sample_weights,
                                                             test_size=args.validation_split,
                                                             random_state=42,
                                                             stratify=labels)

    training_set = MyDataset(train_texts, train_labels, args)
    training_generator = DataLoader(training_set, **training_params)
    model = CharacterLevelCNN(args, number_of_classes)

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.start_lr, momentum=0.9
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.start_lr
        )

    start_lr = args.start_lr
    end_lr = args.end_lr
    lr_find_epochs = args.epochs
    smoothing = args.smoothing

    def lr_lambda(x): return math.exp(
        x * math.log(end_lr / start_lr) / (lr_find_epochs * len(training_generator)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    learning_rates = []

    for epoch in range(lr_find_epochs):
        print(f'[epoch {epoch + 1} / {lr_find_epochs}]')
        progress_bar = tqdm(enumerate(training_generator),
                            total=len(training_generator))
        for iter, batch in progress_bar:
            features, labels = batch
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            learning_rates.append(lr)

            if iter == 0:
                losses.append(loss.item())
            else:
                loss = smoothing * loss.item() + (1 - smoothing) * losses[-1]
                losses.append(loss)

    plt.semilogx(learning_rates, losses)
    plt.savefig('./plots/losses_vs_lr.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Character Based CNN for text classification')
    parser.add_argument('--data_path', type=str,
                        default='./data/train.csv')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--label_column', type=str, default='Sentiment')
    parser.add_argument('--text_column', type=str, default='SentimentText')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--chunksize', type=int, default=50000)
    parser.add_argument('--encoding', type=str, default='utf-8')
    parser.add_argument('--sep', type=str, default=',')
    parser.add_argument('--steps', nargs='+', default=['lower'])
    parser.add_argument('--group_labels', type=str,
                        default=None, choices=[None, 'binarize'])
    parser.add_argument('--ratio', type=float, default=1)

    parser.add_argument('--alphabet', type=str,
                        default='abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"\\/|_@#$%^&*~`+-=<>()[]{}')
    parser.add_argument('--number_of_characters', type=int, default=69)
    parser.add_argument('--extra_characters', type=str, default='')
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str,
                        choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=1)

    parser.add_argument('--start_lr', type=float, default=1e-5)
    parser.add_argument('--end_lr', type=float, default=1e-2)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()
    run(args)
